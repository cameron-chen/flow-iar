import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import normflows as nf
import numpy as np
import torch as th
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from geomloss import SamplesLoss
from gym import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (BaseFeaturesExtractor,
                                                   FlattenExtractor)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import obs_as_tensor
from survae.distributions import (ConditionalMeanNormal,
                                  ConditionalMeanStdNormal, ConditionalNormal)
from survae.flows import ConditionalInverseFlow
from survae.nn.layers import ElementwiseParams
from survae.transforms import (ActNormBijection, ActNormBijection1d,
                               ActNormBijection2d,
                               ConditionalAffineCouplingBijection, Conv1x1,
                               PermuteAxes, Reshape, Reverse, Shuffle)
from torch import nn
from torch.distributions import Uniform

from policies.flow_policy.util import BufferSingleAttr, DictBufferSingleAttr
from utils.distributions import ObsCondBinaryEncoder, ObsCondDiscreteEncoder
from utils.flows.CategoricalNF.flows.autoregressive_coupling import \
    CouplingMixtureCDFCoupling
from utils.flows.CategoricalNF.flows.autoregressive_coupling2 import \
    AutoregressiveMixtureCDFCoupling
from utils.flows.CategoricalNF.networks.autoregressive_layers2 import \
    CouplingLSTMModel
from utils.flows.cond_flow import ConditionalFlow_v2 as ConditionalFlow
from utils.flows.cond_flow import ContextNet, IdxContextNet
from utils.flows.loss import elbo_bpd_cond
from utils.transforms import (AffineCouplingBijection,
                              ConditionalBinaryProductArgmaxSurjection,
                              ConditionalDiscreteArgmaxSurjection, Residual)
from utils.util import op_on_obs
from utils.validator.base import Validator

from .util import ar_func


class ActorCriticFlowPolicy(ActorCriticPolicy):
    """Policy class for flow-based actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.
    
    Args:

    """
    def __init__(
        self, 
        observation_space: gym.spaces.Space, 
        action_space: gym.spaces.Space, 
        lr_schedule: Schedule, 
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None, 
        activation_fn: Type[nn.Module] = nn.Tanh, 
        ortho_init: bool = True, use_sde: bool = False, 
        log_std_init: float = 0.0, 
        full_std: bool = True, 
        sde_net_arch: Optional[List[int]] = None, 
        use_expln: bool = False, 
        squash_output: bool = False, 
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor, 
        features_extractor_kwargs: Optional[Dict[str, Any]] = None, 
        normalize_images: bool = True, 
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam, 
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        flow_type: str = None,
        flow_net_hidden_size: int = 128,
        context_size: int = 64,
        num_flow_layers: int = 4,
        batch_size_flow_updating: int = 64,
        n_iters_flow_pretraining: int = 0,
        val_steps: int = 5,
        val_batch_size: int = 64,
        elbo_steps: int = 1,
        n_samples_prob_est: int = 4,
        pol_grad_G: bool = False,
        elbo_Q: bool = False,
        flow_base_dist: str = 'cond_mean_std_gauss',
        act_encoding_scheme: str = 'one_hot',
        posterior_type: str = 'normal',
        num_posterior_layers: int = 2,
        has_act_corr: bool = False,
        act_corr_prot: str = None,
        lmd_corr: float = 1.0,
        noise_std: float = 0.0,
        rep_reg: int = 125,
        cond_bijection: bool = False,
        sandwich_evidence: bool = False,
        ensemble_mode: str = None,
    ):
        feasible_action_space_cls = [spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary]
        assert any(isinstance(action_space, space_cls) for space_cls in feasible_action_space_cls)
        
        self.num_classes = self._num_classes(action_space)
        self.seq_len =  1
        self.flow_type = flow_type
        self.flow_net_hidden_size = flow_net_hidden_size
        self.context_size = context_size
        self.num_flow_layers = num_flow_layers
        self.batch_size_flow_updating = batch_size_flow_updating
        self.n_iters_flow_pretraining = n_iters_flow_pretraining
        self.val_steps = val_steps
        self.val_batch_size = val_batch_size
        self.elbo_steps = elbo_steps
        self.n_samples_prob_est = n_samples_prob_est
        self.flow_base_dist = flow_base_dist
        self.act_encoding_scheme = act_encoding_scheme # 'one_hot' or 'cartesian_product'
        self.pol_grad_G = pol_grad_G
        self.elbo_Q = elbo_Q
        self.has_act_corr = has_act_corr
        self.act_corr_prot = act_corr_prot
        self.lmd_corr = lmd_corr
        self.noise_std = noise_std
        self.rep_reg = rep_reg
        self.posterior_type = posterior_type
        self.num_posterior_layers = num_posterior_layers
        self.rep_elbo = 50
        self.elbo_threshold = 10
        self.cond_bijection = cond_bijection
        self.max_grad_norm = 1.0
        self.sandwich_evidence = False if ensemble_mode is None else True
        self.ensemble_mode = ensemble_mode

        super().__init__(
            observation_space, 
            action_space, 
            lr_schedule, 
            net_arch, 
            activation_fn, 
            ortho_init, 
            use_sde, 
            log_std_init, 
            full_std, 
            sde_net_arch, 
            use_expln, 
            squash_output, 
            features_extractor_class, 
            features_extractor_kwargs, 
            normalize_images, 
            optimizer_class, 
            optimizer_kwargs
        )
        buffer_cls = DictBufferSingleAttr if isinstance(observation_space, spaces.Dict) else BufferSingleAttr
        self.obs_buffer = buffer_cls(attr_space=observation_space, device=self.device)
        blur = 0.05
        self.wasserstein_loss = SamplesLoss(blur=blur)
        self.san_evi_weight_fn = self._setup_sandwich_evidence_weight() if self.sandwich_evidence \
            else None

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        Args:
            lr_schedule: Learning rate schedule
                lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        # Action net
        self.flow_net, self.flow_pv = self._build_flow_net()

        # Validator net
        if self.has_act_corr and 'val' in self.act_corr_prot: 
            self.validator = Validator(n_cls=self.num_classes, device=self.device, 
                                       c_dim=self.mlp_extractor.latent_dim_pi)

        # Value net
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.flow_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        if self.pol_grad_G:
            # Policy gradient will only update G (action generator)
            #   Freeze the posterior to setup the optimizer
            #   When call optimizer.step(), the policy excluding posterior will be updated
            self._freeze_model(self.posterior)
            self.optimizer = self.optimizer_class(
                filter(lambda p: p.requires_grad, self.parameters()), 
                lr=lr_schedule(1), **self.optimizer_kwargs)
            self._unfreeze_model(self.posterior)
        else: 
            # Policy gradient will update G and Q (action generator and posterior)
            #   When call optimizer.step(), the policy including posterior will be updated
            self.optimizer = self.optimizer_class(
                self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        if self.elbo_Q:
            # ELBO gradient will only update Q (posterior)
            #   When call optimizer_flow_net.step(), only the posterior will be updated
            self.optimizer_flow_net = self.optimizer_class(
                self.posterior.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        else:
            # ELBO gradient will update G and Q (action generator and posterior)
            self.optimizer_flow_net = self.optimizer_class(
                self.flow_net.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        if self.has_act_corr and 'val' in self.act_corr_prot: 
            self.optimizer_validator = self.optimizer_class(
                self.validator.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _build_flow_net(self) -> ConditionalFlow:
        # Preprocessing
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        if self.act_encoding_scheme == 'one_hot':
            argmax_surjection_cls = ConditionalDiscreteArgmaxSurjection
            encoder_cls = ObsCondDiscreteEncoder
            K = self.num_classes
        elif self.act_encoding_scheme == 'cartesian_product':
            argmax_surjection_cls = ConditionalBinaryProductArgmaxSurjection
            encoder_cls = ObsCondBinaryEncoder
            K = argmax_surjection_cls.classes2dims(self.num_classes)
        else:
            raise ValueError(f"Unknown action encoding scheme: {self.act_encoding_scheme}")

        # Transforms
        transforms = []
        if isinstance(self.action_space, spaces.Discrete):
            hidden_dim_encoder = 64
            q_type = self.posterior_type
            coupling_layer_cls = ConditionalAffineCouplingBijection if self.cond_bijection else AffineCouplingBijection

            # Flow - Argmax Surjection
            ## Encoder context
            context_net = ContextNet(
                act_dim=self.num_classes, obs_dim=latent_dim_pi, n_d_act=False,
                hidden_dim=self.flow_net_hidden_size,output_dim=self.context_size,
            )
            ## Encoder base
            encoder_base = ConditionalMeanStdNormal(
                nn.Linear(self.context_size, K),
                scale_shape=(K,),
            )

            ## Encoder transforms
            if q_type == "flow":
                encoder_transforms = []
                encoder_transforms.append(Reshape((1,K), (K,)))
                def net(dim):
                    return nn.Sequential(
                    nn.Linear(-(-dim//2), hidden_dim_encoder), nn.ReLU(),
                    nn.Linear(hidden_dim_encoder, 2*(dim//2)), ElementwiseParams(2))
                for step in range(self.num_posterior_layers):
                    encoder_transforms.append(AffineCouplingBijection(net(K)))
                    encoder_transforms.append(ActNormBijection(K))
                    if step < self.num_posterior_layers-1: 
                        encoder_transforms.append(Reverse(K))
                encoder_transforms.append(Reshape((K,), (1,K)))
            elif q_type == "normal":
                encoder_transforms = [Reshape((1,K), (K,)), Reshape((K,), (1,K))] # (B,1,K)
            else:
                raise ValueError(f"Unknown q_type: {q_type}")
            ## Encoder 
            encoder = encoder_cls(ConditionalInverseFlow(
                base_dist=encoder_base,
                transforms=encoder_transforms,
                context_init=context_net), dims=K
            )
            self.posterior = argmax_surjection_cls(encoder, self.num_classes)
            self.flow_qv = encoder
            transforms.append(self.posterior)
            ## Reshape
            transforms.append(Reshape((1,K),(K,))) # (B,1,K) -> (B,K)

            # Flow - flow layers
            def net():
                if self.cond_bijection:
                    out = []
                    out.append(
                        nn.Sequential(
                            nn.Linear(2*-(-K//2), self.flow_net_hidden_size), nn.ReLU(),
                            nn.Linear(self.flow_net_hidden_size, self.flow_net_hidden_size), nn.ReLU(),
                            nn.Linear(self.flow_net_hidden_size, 2*(K//2)), ElementwiseParams(2))
                    )
                    out.append(nn.Sequential(nn.Linear(latent_dim_pi, -(-K//2)), nn.ReLU()))
                    return out
                else:
                    return [nn.Sequential(
                            nn.Linear(-(-K//2), self.flow_net_hidden_size), nn.ReLU(),
                            nn.Linear(self.flow_net_hidden_size, self.flow_net_hidden_size), nn.ReLU(),
                            nn.Linear(self.flow_net_hidden_size, 2*(K//2)), ElementwiseParams(2))]
            for step in range(self.num_flow_layers):
                transforms.append(coupling_layer_cls(*net()))
                transforms.append(ActNormBijection(K))
                if step < self.num_flow_layers - 1: transforms.append(Reverse(K))
            current_shape = (K, )

            # Regularizing flow 
            transforms_pv = transforms[1:]

            # Base distribution
            if self.flow_base_dist == 'cond_mean_gauss': # Fixed std
                normal_net = nn.Linear(latent_dim_pi, K)
                base_dist = ConditionalMeanNormal(net=normal_net) 
            elif self.flow_base_dist == 'cond_mean_std_gauss': # learnable std
                normal_net = nn.Linear(latent_dim_pi, K)
                base_dist = ConditionalMeanStdNormal(net=normal_net, scale_shape=current_shape) 
            elif self.flow_base_dist == 'cond_gauss': # cond mean and std
                normal_net = nn.Linear(latent_dim_pi, 2*K)
                base_dist = ConditionalNormal(net=normal_net, split_dim=-1) 

        elif isinstance(self.action_space, spaces.MultiDiscrete): 
            # Args
            C = 1
            L = self.seq_len = self.action_space.nvec.shape[0]
            hidden_dim_encoder = 64
            q_type = self.posterior_type
            coupling_layer_cls = ConditionalAffineCouplingBijection if self.cond_bijection else AffineCouplingBijection
            
            if self.flow_type == 'ar':
                actnorm = False
                perm_length = 'reverse'
                perm_channel = 'none'
                lstm_layers = 2
                lstm_size = 256 # 2048 in ArgMax Flow
                input_dp_rate = 0.25

                # Flow - Argmax Surjection
                ## Encoder context
                context_net = IdxContextNet(
                    act_dim=self.num_classes, obs_dim=latent_dim_pi,
                    hidden_dim=self.flow_net_hidden_size,output_dim=self.context_size,)
                ## Encoder base
                encoder_base = ConditionalMeanStdNormal(
                    nn.Conv1d(self.context_size, C*K, kernel_size=1, padding=0),scale_shape=(C*K,L))
                ## Encoder transforms
                encoder_transforms = []
                if q_type == "flow":
                    encoder_transforms.append(Reshape((C*K,L), (C*K*L,))) # (B,C*K,L) -> (B,C*K*L)
                    def encoder_net(dim):
                        return nn.Sequential(
                        nn.Linear(-(-dim//2), hidden_dim_encoder), nn.ReLU(),
                        nn.Linear(hidden_dim_encoder, 2*(dim//2)), ElementwiseParams(2))
                    for step in range(self.num_posterior_layers):
                        encoder_transforms.append(AffineCouplingBijection(encoder_net(C*K*L)))
                        encoder_transforms.append(ActNormBijection(C*K*L))
                        if step < self.num_posterior_layers - 1:
                            encoder_transforms.append(Reverse(C*K*L))
                    encoder_transforms.append(Reshape((C*K*L,), (C,K,L))) # (B,C*K*L) -> (B,C,K,L)
                    encoder_transforms.append(PermuteAxes([0,1,3,2])) # (B,C,K,L) -> (B,C,L,K)
                elif q_type == "normal":
                    encoder_transforms.append(Reshape((C*K,L), (C,K,L))) # (B,C*K*L) -> (B,C,K,L)
                    encoder_transforms.append(PermuteAxes([0,1,3,2])) # (B,C,K,L) -> (B,C,L,K)
                else:
                    raise ValueError(f"Unknown q_type: {q_type}")
                ## Encoder
                encoder = encoder_cls(ConditionalInverseFlow(
                    base_dist=encoder_base,
                    transforms=encoder_transforms,
                    context_init=context_net), dims=K
                )
                self.posterior = argmax_surjection_cls(
                    encoder, self.num_classes, noise_std=self.noise_std)
                transforms.append(self.posterior)
                ## Reshape
                transforms.append(PermuteAxes([0,1,3,2])) # (B,C,L,K) -> (B,C,K,L)
                transforms.append(Reshape((C,K,L), (C*K,L))) # (B,C,K,L) -> (B,C*K,L)
                current_shape = (C*K,L)

                # Flow - flow layers
                for step in range(self.num_flow_layers):
                    if step > 0:
                        if actnorm: transforms.append(ActNormBijection1d(current_shape[0]))
                        if perm_length == 'reverse':    transforms.append(Reverse(current_shape[1], dim=2))
                        if perm_channel == 'conv':      transforms.append(Conv1x1(current_shape[0], slogdet_cpu=False))
                        elif perm_channel == 'shuffle': transforms.append(Shuffle(current_shape[0]))

                    def model_func(c_out):
                        return ar_func(
                            c_in=current_shape[0],
                            c_out=c_out,
                            hidden=lstm_size,
                            num_layers=lstm_layers,
                            max_seq_len=L,
                            input_dp_rate=input_dp_rate)

                    transforms.append(
                        AutoregressiveMixtureCDFCoupling(
                            c_in=current_shape[0],
                            model_func=model_func,
                            block_type="LSTM model",
                            num_mixtures=27)
                    )

            elif self.flow_type == 'shallow_coupling':
                # Flow - Argmax Surjection
                ## Encoder context
                context_net = ContextNet(
                    act_dim=self.num_classes, obs_dim=latent_dim_pi, n_d_act=True,
                    hidden_dim=self.flow_net_hidden_size,output_dim=self.context_size,
                )
                ## Encoder base
                encoder_shape = (C*K,L)
                encoder_base = ConditionalMeanStdNormal(
                    nn.Conv1d(self.context_size, encoder_shape[0], kernel_size=1, padding=0),scale_shape=encoder_shape)
                ## Encoder transforms
                encoder_transforms = []
                if q_type == "flow":
                    encoder_transforms.append(Reshape((C*K,L), (C*K*L,))) # (B,C*K,L) -> (B,C*K*L)
                    def encoder_net(dim):
                        return nn.Sequential(
                        nn.Linear(-(-dim//2), hidden_dim_encoder), nn.ReLU(),
                        nn.Linear(hidden_dim_encoder, 2*(dim//2)), ElementwiseParams(2))
                    for step in range(self.num_posterior_layers):
                        encoder_transforms.append(AffineCouplingBijection(encoder_net(C*K*L)))
                        encoder_transforms.append(ActNormBijection(C*K*L))
                        if step < self.num_posterior_layers - 1:
                            encoder_transforms.append(Reverse(C*K*L))
                    encoder_transforms.append(Reshape((C*K*L,), (C,K,L))) # (B,C*K*L) -> (B,C,K,L)
                    encoder_transforms.append(PermuteAxes([0,1,3,2])) # (B,C,K,L) -> (B,C,L,K)
                elif q_type == "normal":
                    encoder_transforms.append(Reshape((C*K,L), (C,K,L))) # (B,C*K*L) -> (B,C,K,L)
                    encoder_transforms.append(PermuteAxes([0,1,3,2])) # (B,C,K,L) -> (B,C,L,K)
                else:
                    raise ValueError(f"Unknown q_type: {q_type}")
                ## Encoder 
                encoder = encoder_cls(ConditionalInverseFlow(
                    base_dist=encoder_base,
                    transforms=encoder_transforms,
                    context_init=context_net), dims=K
                )
                self.posterior = argmax_surjection_cls(encoder, self.num_classes)
                self.flow_qv = encoder
                transforms.append(self.posterior)
                transforms.append(PermuteAxes([0,1,3,2])) # (B,C,L,K) -> (B,C,K,L)
                transforms.append(Reshape((C,K,L), (C*K*L,))) # (B,C,K,L) -> (B,C*K*L)
                current_shape = (C*K*L,)

                # Flow - flow layers
                def net(dim):
                    if self.cond_bijection:
                        out = []
                        out.append(
                            nn.Sequential(
                            nn.Linear(2*-(-dim//2), self.flow_net_hidden_size), nn.ReLU(),
                            nn.Linear(self.flow_net_hidden_size, self.flow_net_hidden_size), nn.ReLU(),
                            nn.Linear(self.flow_net_hidden_size, 2*(dim//2)), ElementwiseParams(2))
                        )
                        out.append(nn.Sequential(nn.Linear(latent_dim_pi, -(-dim//2)), nn.ReLU()))
                        return out
                    else:
                        return [nn.Sequential(
                            nn.Linear(-(-dim//2), self.flow_net_hidden_size), nn.ReLU(),
                            nn.Linear(self.flow_net_hidden_size, self.flow_net_hidden_size), nn.ReLU(),
                            nn.Linear(self.flow_net_hidden_size, 2*(dim//2)), ElementwiseParams(2))]
                for step in range(self.num_flow_layers):
                    transforms.append(coupling_layer_cls(*net(current_shape[0])))
                    transforms.append(ActNormBijection(current_shape[0]))
                    if step < self.num_flow_layers - 1: transforms.append(Reverse(current_shape[0]))

                transforms.append(Reshape((C*K*L,), (C*K,L))) # (B,C*K*L) -> (B,C*K,L)
                current_shape = (C*K,L)
                
            elif self.flow_type == 'coupling':
                actnorm = False
                perm_length = 'reverse'
                perm_channel = 'conv'
                lstm_layers = 2
                lstm_size = 256 
                input_dp_rate = 0.05
                num_mixtures = 8

                # Flow - Argmax Surjection
                ## Encoder context
                context_net = ContextNet(
                    act_dim=self.num_classes, obs_dim=latent_dim_pi, n_d_act=True,
                    hidden_dim=self.flow_net_hidden_size,output_dim=self.context_size,
                )
                ## Encoder base
                encoder_shape = (C*K, L)
                encoder_base = ConditionalMeanStdNormal(
                    nn.Conv1d(self.context_size, C*K, kernel_size=1, padding=0),scale_shape=(C*K,L))
                ## Encoder transforms
                encoder_transforms = [] 
                if q_type == "flow":
                    encoder_transforms.append(Reshape((C*K,L), (C*K*L,))) # (B,C*K,L) -> (B,C*K*L)
                    def encoder_net(dim):
                        return nn.Sequential(
                        nn.Linear(-(-dim//2), hidden_dim_encoder), nn.ReLU(),
                        nn.Linear(hidden_dim_encoder, 2*(dim//2)), ElementwiseParams(2))
                    for step in range(self.num_posterior_layers):
                        encoder_transforms.append(AffineCouplingBijection(encoder_net(C*K*L)))
                        encoder_transforms.append(ActNormBijection(C*K*L))
                        if step < self.num_posterior_layers - 1:
                            encoder_transforms.append(Reverse(C*K*L))
                    encoder_transforms.append(Reshape((C*K*L,), (C,K,L))) # (B,C*K*L) -> (B,C,K,L)
                    encoder_transforms.append(PermuteAxes([0,1,3,2])) # (B,C,K,L) -> (B,C,L,K)
                elif q_type == "normal":
                    encoder_transforms.append(Reshape((C*K,L), (C,K,L))) # (B,C*K*L) -> (B,C,K,L)
                    encoder_transforms.append(PermuteAxes([0,1,3,2])) # (B,C,K,L) -> (B,C,L,K)
                else:
                    raise ValueError(f"Unknown q_type: {q_type}")
                ## Encoder 
                encoder = encoder_cls(ConditionalInverseFlow(
                    base_dist=encoder_base,
                    transforms=encoder_transforms,
                    context_init=context_net), dims=K
                )
                self.posterior = argmax_surjection_cls(encoder, self.num_classes)
                self.flow_qv = encoder
                transforms.append(self.posterior)
                ## Reshape
                transforms.append(PermuteAxes([0,1,3,2])) # (B,C,L,K) -> (B,C,K,L)
                transforms.append(Reshape((C,K,L), (C*K,L))) # (B,C,K,L) -> (B,C*K,L)
                current_shape = (C*K,L)

                # Flow - flow layers
                for step in range(self.num_flow_layers):
                    if step > 0:
                        if actnorm: transforms.append(ActNormBijection1d(current_shape[0]))
                        if perm_length == 'reverse':    transforms.append(Reverse(current_shape[1], dim=2))
                        if perm_channel == 'conv':      transforms.append(Conv1x1(current_shape[0], slogdet_cpu=False))
                        elif perm_channel == 'shuffle': transforms.append(Shuffle(current_shape[0]))

                    def model_func(c_in, c_out):
                        return CouplingLSTMModel(
                                    c_in=c_in,
                                    c_out=c_out,
                                    max_seq_len=L,
                                    num_layers=lstm_layers,
                                    hidden_size=lstm_size,
                                    dp_rate=0,
                                    input_dp_rate=input_dp_rate)

                    transforms.append(
                        CouplingMixtureCDFCoupling(
                            c_in=K//2,
                            c_out=K//2+K%2,
                            model_func=model_func,
                            block_type="LSTM model",
                            num_mixtures=num_mixtures)
                    )

            elif self.flow_type == 'res':
                import warnings
                warnings.warn("Metric flow_net_elbo is not accurate for `Residual`, "+\
                    "refer to https://github.com/rtqichen/residual-flows#density-estimation-experiments for details.")

                actnorm = True 
                channels = [1, 512, 512, 1]
                kernel_size = [3, 1, 3]
                lipschitz_const = 0.98
                max_lipschitz_iter = None
                lipschitz_tolerance = 0.001

                 # Flow - Argmax Surjection
                ## Encoder context
                context_net = ContextNet(
                    act_dim=self.num_classes, obs_dim=latent_dim_pi, n_d_act=True,
                    hidden_dim=self.flow_net_hidden_size,output_dim=self.context_size,
                )
                ## Encoder base
                encoder_shape = (C*K, L)
                encoder_base = ConditionalMeanStdNormal(
                    nn.Conv1d(self.context_size, C*K, kernel_size=1, padding=0),scale_shape=(C*K,L))
                ## Encoder transforms
                encoder_transforms = []
                if q_type == "flow":
                    encoder_transforms.append(Reshape((C*K,L), (C*K*L,))) # (B,C*K,L) -> (B,C*K*L)
                    def encoder_net(dim):
                        return nn.Sequential(
                        nn.Linear(-(-dim//2), hidden_dim_encoder), nn.ReLU(),
                        nn.Linear(hidden_dim_encoder, 2*(dim//2)), ElementwiseParams(2))
                    for step in range(self.num_posterior_layers):
                        encoder_transforms.append(AffineCouplingBijection(encoder_net(C*K*L)))
                        encoder_transforms.append(ActNormBijection(C*K*L))
                        if step < self.num_posterior_layers - 1:
                            encoder_transforms.append(Reverse(C*K*L))
                    encoder_transforms.append(Reshape((C*K*L,), (C,K,L))) # (B,C*K*L) -> (B,C,K,L)
                    encoder_transforms.append(PermuteAxes([0,1,3,2])) # (B,C,K,L) -> (B,C,L,K)
                elif q_type == "normal":
                    encoder_transforms.append(Reshape((C*K,L), (C,K,L))) # (B,C*K*L) -> (B,C,K,L)
                    encoder_transforms.append(PermuteAxes([0,1,3,2])) # (B,C,K,L) -> (B,C,L,K)
                else:
                    raise ValueError(f"Unknown q_type: {q_type}")
                ## Encoder 
                encoder = encoder_cls(ConditionalInverseFlow(
                    base_dist=encoder_base,
                    transforms=encoder_transforms,
                    context_init=context_net), dims=K
                )
                self.posterior = argmax_surjection_cls(encoder, self.num_classes)
                transforms.append(self.posterior)

                # Flow - flow layers
                for step in range(self.num_flow_layers):
                    nnet = nf.nets.LipschitzCNN(
                        channels=channels, kernel_size=kernel_size,
                        lipschitz_const=lipschitz_const, max_lipschitz_iter=max_lipschitz_iter,
                        lipschitz_tolerance=lipschitz_tolerance
                    )
                    transforms.append(Residual(nnet, reverse=False))
                    if actnorm: transforms.append(ActNormBijection2d(C))
                transforms.append(PermuteAxes([0,1,3,2])) # (B,C,L,K) -> (B,C,K,L)
                transforms.append(Reshape((C,K,L), (C*K,L))) # (B,C,K,L) -> (B,C*K,L)
                current_shape = (C*K,L)

            # Regularizing flow
            transforms_pv = transforms[1:]

            # Base distribution
            if self.flow_base_dist == 'cond_mean_gauss': # Fixed std
                normal_net = nn.Sequential(
                    nn.Linear(latent_dim_pi, np.prod(current_shape)),
                    Rearrange("b (kc l) -> b kc l", l=current_shape[1])
                )
                base_dist = ConditionalMeanNormal(net=normal_net) 
            elif self.flow_base_dist == 'cond_mean_std_gauss': # learnable std
                normal_net = nn.Sequential(
                    nn.Linear(latent_dim_pi, np.prod(current_shape)),
                    Rearrange("b (kc l) -> b kc l", l=current_shape[1])
                )
                base_dist = ConditionalMeanStdNormal(net=normal_net, scale_shape=current_shape) 
            else:
                raise NotImplementedError(f"Base distribution {self.flow_base_dist} not implemented")

        elif isinstance(self.action_space, spaces.MultiBinary):
            raise NotImplementedError('MultiBinary is not supported yet. Please converted action space to Discrete.')

        else:
            raise RuntimeError(f'Unsupported action space: {self.action_space}.')

        return ConditionalFlow(base_dist=base_dist,  transforms=transforms), \
            ConditionalFlow(base_dist=base_dist,  transforms=transforms_pv)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        Args:
            obs: Observation
            deterministic: Whether to sample or use deterministic actions
        
        Return: 
            action, value and log probability of the action

        Note: Flow policy has no deterministic mode
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        actions = self.flow_net.sample(latent_pi)
        log_prob = self._log_prob_by_samples(actions, latent_pi, self.n_samples_prob_est)
        actions = actions.squeeze(1)

        return actions, values, log_prob

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # Note: obs and actions are leaf nodes

        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        log_prob = self._log_prob_by_samples(
            rearrange(actions.long(), "b ... -> b () ..."), 
            latent_pi, self.n_samples_prob_est)
        values = self.value_net(latent_vf)
        # Hack: no analytical form, output entropy=None, resulting in entropy approaximation
        return values, log_prob, None

    def reg_ref_dist(self, obs: th.Tensor) -> th.Tensor:
        """compute a distance between reference distribution and flow distribution"""
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        sample_pv = self.flow_pv.sample_enable_grad(latent_pi).flatten(1)

        samp_size, shape = sample_pv.size(0), sample_pv.size()[1:]
        # ref_dist = Normal(
        #     loc=th.zeros(shape, device=self.device),
        #     scale=th.ones(shape, device=self.device)
        # )
        ref_dist = Uniform(
            -1*th.ones(shape, device=self.device),
            th.ones(shape, device=self.device)
        )
        sample_ref = ref_dist.sample(th.Size([samp_size]))
        loss = self.wasserstein_loss

        assert sample_pv.requires_grad, 'sample_pv should require grad'
        return loss(sample_pv, sample_ref)

    def update_obs_buffer(self, obs: Union[np.ndarray, Dict[str, np.ndarray]]):
        if isinstance(obs, dict):
            obs_reshaped = {}
            for k, v in obs.items():
                assert len(v.shape) >= 3
                obs_reshaped[k] = rearrange(v.copy(), "s b ... -> (b s) ...")
            self.obs_buffer.add(obs_reshaped)
        elif isinstance(obs, np.ndarray):
            assert len(obs.shape) >= 3
            self.obs_buffer.add(rearrange(obs.copy(), "s b ... -> (b s) ..."))
        else:
            raise RuntimeError(f"Unsupported observation type: {type(obs)}")

    def flow_net_updating(self, obs: th.Tensor):
        """Update the flow net before it can create proper approximated log_prob.
        """
        with th.no_grad():
            # Preprocess the observation if needed
            features = self.extract_features(obs)
            latent_pi, _ = self.mlp_extractor(features)

            # Generate actions: 
            #   generate several batches of actions conditioned on the same context,
            #   such that actions are drawn from the same distribution.
            actions = []
            for step in range(self.elbo_steps):
                actions.append(th.clone(self.flow_net.sample(latent_pi)))

        # ELBO updating
        for act in actions:
            loss = self._elbo_updating(act, latent_pi)
        
        return loss
    
    def turn_on_explore_mode(self):
        return self.posterior.turn_on_explore_mode()

    def turn_off_explore_mode(self):
        return self.posterior.turn_off_explore_mode()
    
    @property
    def explore_mode(self):
        return self.posterior.explore_mode
    
    def set_noise_eps(self, eps: float):
        return self.posterior.set_noise_eps(eps)


    def act_corr_val(
        self, 
        obs: th.Tensor,
        val_inv_act_gen_fn: Callable[[np.ndarray], Any],
        act_check_fn: Callable[[np.ndarray], Any],
    )-> Tuple[float, float]:
        """
        Action correction step: update the validator, after which updates the flow net
            by the updated validator. 
        """
        raise NotImplementedError(
            "Action correction step has not completed implementation for this class. \n"
            "Need to handle the explore_mode. "
        )
        # Optimize V
        ## Create dataset
        x, y, c = val_inv_act_gen_fn(obs.data.cpu().numpy())
        val_loss = []
        idx = np.random.permutation(x.shape[0])
        start_pos = 0
        ## Perform optimization
        for _ in range(self.val_steps):
            idx_batch = idx[start_pos:start_pos+self.val_batch_size]
            start_pos += self.val_batch_size
            x_batch, y_batch, c_batch = [obs_as_tensor(o, self.device) for o in [x[idx_batch], y[idx_batch], c[idx_batch]]]
            # will not update features_extractor and mlp_extractor
            with th.no_grad(): latent_c_batch,_ = self.mlp_extractor(self.extract_features(c_batch)) 
            val_loss.append(self._validator_updating(x_batch, y_batch, latent_c_batch))

        # Optimize flow net
        flow_loss = self._validator_correcting(obs, act_check_fn, self.val_batch_size)

        return np.array(val_loss).mean().item(), flow_loss
    
    def act_corr_flow(
        self,
        obs: th.Tensor,
        act_check_fn: Callable[[np.ndarray], Any],
        mini_batch_size: int = 32,
        n_samples: int = 3,
    )-> None:
        raise NotImplementedError(
            "Action correction step has not completed implementation for this class. \n"
            "Need to handle the explore_mode. "
        )
        assert mini_batch_size < self.val_batch_size
        assert not obs.requires_grad

        # Extend obs
        # obs_n_repeat = repeat(obs, "b ... -> (b r) ...", r=n_samples)
        obs_n_repeat = op_on_obs(repeat, obs, pattern="b ... -> (b r) ...", r=n_samples)

        # Generate and sample val/inv actions
        with th.no_grad(): latent_pi,_ = self.mlp_extractor(self.extract_features(obs_n_repeat))
        act = self.flow_net.sample(latent_pi)
        val_idx = np.array([act_check_fn(a, o) for a, o 
                             in zip(act.squeeze(1).data.cpu().numpy(), obs_n_repeat.data.cpu().numpy())])
        inv_idx = ~val_idx
        idx = [val_idx, inv_idx]

        for i, il in enumerate(idx):
            if il.sum()>self.val_batch_size:
                idx[i] = np.random.choice(np.where(il)[0], self.val_batch_size, replace=False)
            elif il.sum()>=mini_batch_size:
                idx[i] = np.random.permutation(np.where(il)[0])
            else:
                idx[i] = None
        val_idx, inv_idx = idx

        # Optimize flow net
        if inv_idx is not None:
            act_inv = act[inv_idx]
            latent_pi_inv = latent_pi[inv_idx]
            log_prob_inv = self._log_prob_by_samples(act_inv, latent_pi_inv, self.n_samples_prob_est)
            prob = th.exp(log_prob_inv)
            label = th.full(prob.shape, 0.0, dtype=th.float, device=self.device)
            # Minimize the log prob
            loss = self.lmd_corr*F.binary_cross_entropy(prob, label, reduction="mean")
            
            self.optimizer_flow_net.zero_grad()
            loss.backward()
            self.optimizer_flow_net.step()

    def inv_act_actprob(
        self,
        obs: th.Tensor, 
        act_check_fn: Callable[[np.ndarray], Any],
        mini_batch_size: int = 32,
        n_samples: int = 3,
    )-> Tuple[Optional[th.Tensor], Optional[th.Tensor]]:
        """Generate invalid actions and compute their log prob.

        Returns:
            act_inv (Optional): invalid actions
            log_p_a_inv (Optional): log prob of invalid actions
            latent_pi_inv (Optional): latent representation of observations

        Return None if invalid actions are very few.
        """
        raise NotImplementedError(
            "Action correction step has not completed implementation for this class. \n"
            "Need to handle the explore_mode. "
        )
        assert mini_batch_size < self.val_batch_size
        assert isinstance(obs, th.Tensor)

        # Extend obs
        # obs_n_repeat = repeat(obs, "b ... -> (b r) ...", r=n_samples)
        obs_n_repeat = op_on_obs(repeat, obs, pattern="b ... -> (b r) ...", r=n_samples)

        # Generate and sample val/inv actions
        with th.no_grad(): latent_pi,_ = self.mlp_extractor(self.extract_features(obs_n_repeat))
        act = self.flow_net.sample(latent_pi)
        val_idx = np.array([act_check_fn(a, o) for a, o 
                             in zip(act.squeeze(1).data.cpu().numpy(), obs_n_repeat.data.cpu().numpy())])
        inv_idx = ~val_idx
        idx = [val_idx, inv_idx]

        for i, il in enumerate(idx):
            if il.sum()>self.val_batch_size:
                idx[i] = np.random.choice(np.where(il)[0], self.val_batch_size, replace=False)
            elif il.sum()>=mini_batch_size:
                idx[i] = np.random.permutation(np.where(il)[0])
            else:
                idx[i] = None
        val_idx, inv_idx = idx

        if inv_idx is not None:
            act_inv = act[inv_idx]
            latent_pi_inv = latent_pi[inv_idx]
            log_p_a_inv = self._log_prob_by_samples(act_inv, latent_pi_inv, self.n_samples_prob_est)
            return act_inv.squeeze(1), log_p_a_inv, latent_pi_inv
        else: 
            return None, None, None

    def _elbo_updating(self, action: th.Tensor, context: th.Tensor):
        assert not action.requires_grad
        assert not context.requires_grad
        
        # Compute the ELBO
        loss = elbo_bpd_cond(
            self.flow_net, 
            action.to(self.device),
            context)
        
        # Optimization step
        self.optimizer_flow_net.zero_grad()
        loss.backward()

        # Clip grad norm
        th.nn.utils.clip_grad_norm_(self.flow_net.parameters(), self.max_grad_norm)
        self.optimizer_flow_net.step()
        
        return loss

    def _validator_updating(self, x: th.Tensor, y: th.Tensor, c: th.Tensor)-> float:
        """
        Update the validator by a batch of data.

        Args:
            x: input
            y: label
            c: context
        """
        assert not x.requires_grad
        assert not y.requires_grad
        
        # Compute the loss
        self.optimizer_validator.zero_grad()
        y_p = self.validator.probs(x, c).flatten()
        loss = F.binary_cross_entropy(y_p, y.to(th.float32), reduction='mean')
        # Backprop and update
        loss.backward()
        self.optimizer_validator.step()
        
        return loss.item()
    
    def _validator_correcting(
        self, 
        obs: th.Tensor, 
        act_check_fn: Callable[[np.ndarray], Any],
        batch_size: int,
        n_samples: int = 3,
        min_batch_size: int = 64,
    )-> float:
        """
        Update the flow net by the updated validator.
        """
        assert not obs.requires_grad

        # Extend obs
        # obs_n_repeat = repeat(obs, "b ... -> (b r) ...", r=n_samples)
        obs_n_repeat = op_on_obs(repeat, obs, pattern="b ... -> (b r) ...", r=n_samples)

        # Generate and sample invalid actions
        with th.no_grad(): latent_pi,_ = self.mlp_extractor(self.extract_features(obs_n_repeat))
        act, act_soft = self.flow_net.sample_softmax(latent_pi)
        act, act_soft = act.squeeze(1), act_soft.squeeze(1)
        inv_idx = ~np.array([act_check_fn(a, o) for a, o 
                             in zip(act.data.cpu().numpy(), obs_n_repeat.data.cpu().numpy())])
        
        ## if inv_idx is more than batch_size, then sample from inv_idx
        if inv_idx.sum() > batch_size:
            inv_idx = np.random.choice(np.where(inv_idx)[0], batch_size, replace=False)
        ## elif inv_idx is more than minibatch_size, then permute inv_idx
        elif inv_idx.sum() >= min_batch_size:
            inv_idx = np.random.permutation(np.where(inv_idx)[0])
        ## else, return None
        else:
            return None

        act_soft_inv = act_soft[inv_idx]
        latent_pi_inv = latent_pi[inv_idx]

        # Optimize flow net
        y_p = self.validator.probs(act_soft_inv, latent_pi_inv).flatten() 
        loss = self.lmd_corr * F.binary_cross_entropy(y_p, th.ones_like(y_p).to(self.device), reduction='mean') # loss measure policy's ability to fool validator

        self.optimizer_flow_net.zero_grad()
        loss.backward()
        self.optimizer_flow_net.step()
        
        return loss.item()

    def _log_prob_by_samples(
        self, 
        action: th.Tensor, 
        context: th.Tensor,
        n_samples: int = 4
    ):
        """Estimate the action probability by samples.

        Returns:
            log_prob: log probability of the action
        """
        assert not action.requires_grad
        assert action.dim() > 1

        threshold = -1e2
        alpha = 2.0

        # Extend action & context
        action_n_repeat = repeat(action, 'b ... -> (b r) ...', r=n_samples)
        context_n_repeat = repeat(context, 'b ... -> (b r) ...', r=n_samples)
        # Obtain log_prob
        if self.sandwich_evidence:
            # estimation := (elbo + cubo) / 2
            sample_qv, logq = self.flow_qv.sample_with_log_prob(action_n_repeat, context_n_repeat)
            logp = self.flow_pv.log_prob(sample_qv, context_n_repeat) # (B*n_samples,)
            logw = self._extreme_value_to_nan((logp-logq).view(-1, n_samples), threshold=threshold)
            log_lower = reduce(logw, 'b r -> b', th.nanmean) # (B,), E_{q(v|s)}[log p(v)/q(v|s)]
            log_upper = self._cubo(logp, logq, rep_cubo=n_samples, alpha=alpha) # (B,), 1/n * log E_{q(v|s)}[(p(v)/q(v|s))^alpha]
            log_prob = self._ensemble_estimate(log_lower, log_upper) # (B,)
        else: 
            # estimation := elbo
            log_prob_n_samples = self.flow_net.log_prob(action_n_repeat, context_n_repeat) # (B*n_samples,)
            log_prob_n_samples = self._extreme_value_to_nan(log_prob_n_samples.view(-1, n_samples),
                                                            threshold=threshold)    # (B, n_samples)
            log_prob = reduce(log_prob_n_samples, 'b r -> b', th.nanmean) # (B,)
        return log_prob
        
    @staticmethod
    def _extreme_value_to_nan(
        x: th.Tensor, 
        threshold: float, 
        gt: bool=True
    ):
        """Replace extreme values to nan.

        Args: 
            gt: if True, valid values should be greater than threshold.
                In other words, values less than threshold will be replaced to nan.
        """
        assert x.dim() == 2

        valid_idx = (x>threshold) if gt else (x<threshold)
        valid_idx = valid_idx & (th.isnan(x)==False)
        n_valid = valid_idx.sum(1)

        # all values are within threshold
        if (n_valid == x.shape[1]).all():
            return x
        # at least one value is within threshold
        elif (n_valid > 0).all():
            x = th.where(valid_idx, x, th.tensor(float('nan'), device=x.device))
            if (n_valid/x.shape[1] < 0.5).any():
                warnings.warn("Too many values are replaced by nan for certain states.", RuntimeWarning)
            return x
        # no value is within threshold
        else:
            if gt:
                # should greater than threshold. When no value is, larger the better
                #   replace all nan to -inf
                _x = th.where(th.isnan(x), th.tensor(float('-inf'), device=x.device), x)
                #   take argmax
                _x_max_idx = _x.argmax(1)
                #   allow to use max value though it is less than threshold
                row_idx = th.where(n_valid==0)[0]
                col_idx = _x_max_idx[row_idx]
                valid_idx[row_idx, col_idx] = True
            else:
                # should less than threshold. When no value is, smaller the better
                #   replace all nan to inf
                _x = th.where(th.isnan(x), th.tensor(float('inf'), device=x.device), x)
                #   take argmin
                _x_min_idx = _x.argmin(1)
                #   allow to use min value though it is greater than threshold
                row_idx = th.where(n_valid==0)[0]
                col_idx = _x_min_idx[row_idx]
                valid_idx[row_idx, col_idx] = True
            x = th.where(valid_idx, x, th.tensor(float('nan'), device=x.device))
            warnings.warn("No values are within threshold for certain states.", RuntimeWarning)
            return x

    def _cubo(self, logp, logq, rep_cubo, alpha=2.0, eps=1e-20):
        """Compute the cubo loss.
        """
        logw = (logp - logq).view(-1, rep_cubo) # (B, rep_cubo)
        logw = self._extreme_value_to_nan(logw, threshold=1, gt=False)
        cubo = 1/alpha * (logw.exp().pow(alpha).nanmean(dim=1)+eps).log() # (B,)
        return cubo

    def _ensemble_estimate(self, elbo, cubo):
        """Compute the ensemble estimate.
        """
        mode = self.ensemble_mode
        alpha_fn = self.san_evi_weight_fn
        _cubo = th.clamp(cubo, min=elbo) # log_upper cannot be lower than log_lower

        if mode == 'mean':
            return (elbo + _cubo)/2
        elif mode == 'learned_weight':
            alpha = alpha_fn
        elif mode == 'cond_weight_2':
            combined = th.stack([elbo, _cubo], dim=1) # (B, 2)
            alpha = alpha_fn(combined).squeeze() # (B,)
        elif 'cond_weight_4' in mode:
            combined = th.stack([elbo, _cubo, elbo-_cubo, elbo+_cubo], dim=1) # (B, 4)
            alpha = alpha_fn(combined).squeeze() # (B,)
        else:
            raise NotImplementedError

        return alpha*elbo + (1-alpha)*_cubo
    
    def _setup_sandwich_evidence_weight(self):
        mode = self.ensemble_mode

        if mode == 'mean':
            return None
        elif mode == 'learned_weight':
            # weight shared over samples
            return nn.parameter.Parameter(th.tensor(0.5))
        elif mode == 'cond_weight_2':
            # weight conditioned on samples
            return nn.Sequential(
                nn.BatchNorm1d(2), 
                nn.Linear(2, 32), nn.BatchNorm1d(32), nn.GELU(),
                nn.Linear(32, 1), nn.Sigmoid()
            )
        elif mode == 'cond_weight_4':
            return nn.Sequential(
                nn.BatchNorm1d(4),
                nn.Linear(4, 32), nn.BatchNorm1d(32), nn.GELU(),
                nn.Linear(32, 1), nn.Sigmoid()
            )
        elif mode == 'cond_weight_4_variant':
            # no batch norm in the first layer
            return nn.Sequential(
                nn.Linear(4, 32), nn.BatchNorm1d(32), nn.GELU(),
                nn.Linear(32, 1), nn.Sigmoid()
            )
        else:
            raise NotImplementedError

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Note: Flow policy has no deterministic mode
        """
        features = self.extract_features(observation)
        latent_pi = self.mlp_extractor.forward_actor(features)
        actions = self.flow_net.sample(latent_pi)
        return actions.squeeze(1)
    
    def _freeze_model(self, model: nn.Module):
        for param in model.parameters():
            param.requires_grad = False
    
    def _unfreeze_model(self, model: nn.Module):
        for param in model.parameters():
            param.requires_grad = True

    def _num_classes(self, action_space: spaces.Space) -> int:
        if isinstance(action_space, spaces.Discrete):
            return action_space.n
        elif isinstance(action_space, spaces.MultiDiscrete):
            return action_space.nvec[0]
        elif isinstance(action_space, spaces.MultiBinary):
            return 2
        else:
            raise ValueError(f'Not support action space: {action_space}')
