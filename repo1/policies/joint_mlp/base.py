import itertools
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.distributions import (
    CategoricalDistribution, Distribution, MultiCategoricalDistribution)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (BaseFeaturesExtractor,
                                                   FlattenExtractor)
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from utils.distributions.common import CategoricalMaskDistribution


class ActorCriticJointPolicy(ActorCriticPolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction)
        with joint policy support. Used by A2C, PPO and the likes.
    
    Args:

    """
    def __init__(
        self, 
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
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
    ):
        feasible_action_space_cls = [spaces.Discrete, spaces.MultiDiscrete]
        assert any(isinstance(action_space, space_cls) for space_cls in feasible_action_space_cls)

        self.cand_act = np.array(list(itertools.product(*[range(dim) for dim in action_space.nvec])))

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
    
    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        Args:
            lr_schedule: Learning rate schedule
                lr_schedule(1) is the initial learning rate
        """ 
        self._build_mlp_extractor()
        
        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, MultiCategoricalDistribution): 
            # HACK: update action distribution to support joint policies
            self.action_dist = CategoricalDistribution(np.prod(self.action_dist.action_dims).item()) 
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, CategoricalDistribution): 
            # do nothing for discrete action
            pass
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        Args:
            obs: Observation
            deterministic: Whether to sample or use deterministic actions
            
        Returns:
            action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return self._idx2prod(actions), values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
            given the observations.
        """
        actions = self._prod2idx(actions)
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        features = self.extract_features(observation)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._idx2prod(self._get_action_dist_from_latent(latent_pi).
                    get_actions(deterministic=deterministic))

    # def _get_action_dist_from_latent(self, latent_pi: th.Tensor, masks: th.Tensor) -> Distribution:
    #     """
    #     Retrieve action distribution given the latent codes.

    #     Args: 
    #         latent_pi: Latent code for the actor
        
    #     Returns: 
    #         Action distribution
    #     """
    #     mean_actions = self.action_net(latent_pi)

    #     if isinstance(self.action_dist, (CategoricalMaskDistribution)):
    #         # Here mean_actions are the logits before the softmax
    #         return self.action_dist.proba_distribution(action_logits=mean_actions, masks=masks)
    #     else:
    #         raise ValueError("Invalid action distribution")

    def _prod2idx(self, a_prod: th.Tensor) -> th.Tensor:
        """Convert actions from product form to index form."""
        assert a_prod.ndim == 2
        mapping = lambda x: np.where(np.all(self.cand_act == x, axis=-1))[0]
        return th.from_numpy(np.apply_along_axis(
            mapping, 1, a_prod.data.cpu().numpy())).to(self.device).long().flatten()
    
    def _idx2prod(self, a_idx: th.Tensor) -> th.Tensor:
        """Convert actions from index form to product form."""
        assert a_idx.ndim == 1
        return th.from_numpy(self.cand_act).to(self.device)[a_idx]
