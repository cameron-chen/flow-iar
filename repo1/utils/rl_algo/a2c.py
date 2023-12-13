from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
import torch.nn.functional as F
from einops import rearrange, repeat
from gym import spaces
from joblib import Parallel, delayed
from stable_baselines3 import A2C
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from policies.flow_policy.util import (ObjActDictRolloutBuffer,
                                       ObjActRolloutBuffer)


class A2C_Flow(A2C):

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        min_reg_coef: float = 0.0,
        max_reg_coef: float = 0.1,
        reg_threshold: float = 0.4,
        reg_coef_q: float = 0.0,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(policy, env, learning_rate, n_steps, gamma, gae_lambda, ent_coef, vf_coef, 
                         max_grad_norm, rms_prop_eps, use_rms_prop, use_sde, sde_sample_freq, normalize_advantage, 
                         tensorboard_log, create_eval_env, policy_kwargs, verbose, seed, device, _init_setup_model)
        self.reg_loss_history = []
        self.history_length = 5
        self.reg_coef_1 = None
        self.reg_coef_2 = reg_coef_q
        self.min_reg_coef = min_reg_coef
        self.max_reg_coef = max_reg_coef
        self.sche_freq = 3
        self.reg_threshold = reg_threshold

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer 
            (one gradient step over whole data).
        Introduce regularizing flow. 
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # Update regularizer coeficient
        self._schedule_reg_coef(self.sche_freq)

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            policy_loss = -(advantages * log_prob).mean()

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values)

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            # Regularizing flow
            reg_loss_1 = self._reg_loss_w_base(rollout_data.observations)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + \
                self.reg_coef_1 * reg_loss_1 

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        hist_len = self.history_length
        self.reg_loss_history.append(reg_loss_1.item())
        self.reg_loss_history = self.reg_loss_history[-hist_len:]

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/reg_loss_w_base", reg_loss_1.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    def _reg_loss_w_base(self, obs):
        """
        Compute regularizer loss between the reference N(0,1) and the flow 
            distribution p(v) (marginalize over s).
        """
        if self.reg_coef_1 == 0.0:
            return th.tensor([0.0], device=self.device)
        rep = self.policy.rep_reg
        obs_ext = op_on_obs(repeat, obs, pattern="b ... -> (b r) ...", r=rep)
        return self.policy.reg_ref_dist(obs_ext)
    
    def _reg_loss_w_q(self, obs):
        """
        Compute regularizer loss between the posterior q(v|s) and the flow 
            distribution p(v|s).
        """
        if self.reg_coef_2 == 0.0:
            return th.tensor([0.0], device=self.device)
        rep = self.policy.rep_wasserstein
        obs_ext = op_on_obs(repeat, obs, pattern="b ... -> (b r) ...", r=rep)
        return self.policy.reg_spt(obs_ext)

    def _reg_elbo(self, obs):
        """
        Compute the ELBO of the evidence log_p(a|s).
        """
        rep = self.policy.rep_wasserstein
        return self.policy.reg_elbo(obs, rep)
    
    def _schedule_reg_coef(self, sche_freq=5):
        """
        Schedule the regularizer coefficient.
        """
        self.reg_coef_1 = self.max_reg_coef
        self.logger.record("train/reg_coef_w_base", self.reg_coef_1)
        return

        raise NotImplementedError
        if self._n_updates < 200:
            self.reg_coef_1 = self.max_reg_coef
            self.logger.record("train/reg_coef_w_base", self.reg_coef_1)
            return
        
        reg_hist = self.reg_loss_history
        min_coef = self.min_reg_coef
        max_coef = self.max_reg_coef
        reg_thrs = self.reg_threshold
        if self._n_updates%sche_freq == 0:
            if np.array(reg_hist).mean()>reg_thrs:
                self.reg_coef_1 = max_coef
            elif np.array(reg_hist).mean()>0.5*reg_thrs:
                self.reg_coef_1 = 0.5 * max_coef
            elif np.array(reg_hist).mean()>0.3*reg_thrs:
                self.reg_coef_1 = 0.05 * max_coef
            elif np.array(reg_hist).mean()>0.2*reg_thrs:
                self.reg_coef_1 = 0.03 * max_coef
            else:
                self.reg_coef_1 = min_coef

        self.logger.record("train/reg_coef_w_base", self.reg_coef_1)

class A2C_Corr(A2C):
    
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        ac_coef: float = 0.0,
        loss_type: str = 'prob_inv',
    ):
        super().__init__(policy, env, learning_rate, n_steps, gamma, gae_lambda, ent_coef, vf_coef, 
                         max_grad_norm, rms_prop_eps, use_rms_prop, use_sde, sde_sample_freq, normalize_advantage, 
                         tensorboard_log, create_eval_env, policy_kwargs, verbose, seed, device, _init_setup_model)
        
        self.ac_coef = ac_coef
        self.loss_type = loss_type
    
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer 
            (one gradient step over whole data).

            Introduce action correction loss term.
        """

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            policy_loss = -(advantages * log_prob).mean()

            # Action correction loss
            act_inv, log_prob_inv, context = self.policy.inv_act_actprob(
                rollout_data.observations, self.env.envs[0].act_check)
            act_corr_loss = self.compute_act_corr_loss(act_inv, log_prob_inv, 
                context, loss_type=self.loss_type)

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values)

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.ac_coef * act_corr_loss

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        self.logger.record("train/act_corr_loss", act_corr_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    def compute_act_corr_loss(self, act_inv, log_prob_inv, context, loss_type='prob_inv'):
        """
        Compute action correction loss.

        Returns:
            loss_type: 'prob_inv' or 'validator'
        """
        if act_inv is None:
            return th.tensor(0.0).to(self.device)
        if loss_type == 'prob_inv':
            prob = th.exp(log_prob_inv)
            label = th.full(prob.shape, 0.0, dtype=th.float, device=self.device)
            # Minimize log_prob of invalid actions
            act_corr_loss = F.binary_cross_entropy(prob, label, reduction='mean')
        elif loss_type == 'validator':
            raise RuntimeError("Validator is not trained for joint mode. Implement before using.")

            y_p = self.policy.validator.probs(act_inv, context).flatten()
            act_corr_loss = F.binary_cross_entropy(y_p, th.ones_like(y_p).to(self.device), reduction='mean')
        else:
            raise ValueError('Invalid loss type.')
        
        return act_corr_loss

class A2C_Rej(A2C):

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_reg_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        min_n_samples_inv_rej: int = 8,
        n_jobs: int = -1,
        noise_eps: float = 0.0,
        grad_corr: bool = True,
    ):
        super().__init__(policy, env, learning_rate, n_steps, gamma, gae_lambda, ent_coef, vf_coef, 
                         max_grad_norm, rms_prop_eps, use_rms_prop, use_sde, sde_sample_freq, normalize_advantage, 
                         tensorboard_log, create_eval_env, policy_kwargs, verbose, seed, device, _init_setup_model)
        
        self.act_check_fn = env.envs[0].act_check
        self.n_samples_inv_rej = 64
        self.min_n_samples_inv_rej = min_n_samples_inv_rej
        self.enable_parallel = True if n_jobs > 1 else False
        if self.enable_parallel: self.parallel = Parallel(n_jobs=n_jobs)
        self.noise_eps = noise_eps
        self.reg_coef_1 = max_reg_coef 
        self.grad_corr = grad_corr

        self._valid_sample_rate = 1
        self._last_update_sample_size_timestep = 0
                        
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer 
            (one gradient step over whole data).

            Introduce policy gradient correction term.
        """

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            log_prob_val_sum, num_val_samples, num_samples = self._process_action_samples(rollout_data)
            if self.grad_corr:
                # Policy gradient correction term
                intermediate_vars = log_prob - num_samples/th.pow(num_val_samples, 2) * log_prob_val_sum
            else:
                # Standard policy gradient
                intermediate_vars = log_prob
            policy_loss = -(advantages * intermediate_vars).mean()

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values)

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            # Regularizing flow
            reg_loss_1 = self._reg_loss_w_base(rollout_data.observations)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + \
                self.reg_coef_1 * reg_loss_1

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        self._valid_sample_rate = (num_val_samples/num_samples).mean().item()
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        self.logger.record("train/val_rate_sample", self._valid_sample_rate)
        self.logger.record("train/n_samples", self.n_samples_inv_rej)
        self.logger.record("train/reg_coef_w_base", self.reg_coef_1)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    def collect_rollouts(
        self, 
        env: VecEnv, 
        callback: BaseCallback, 
        rollout_buffer: RolloutBuffer, 
        n_rollout_steps: int
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
            The term rollout here refers to the model-free notion and should not
            be used with the concept of rollout used in model-based RL or planning.

            Collect rollouts for invalid action rejection algorithm.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        # Update training parameters:
        #   Sample size for invalid action rejection algorithm
        #   Noise epsilon
        self._update_sample_size(n_min=self.min_n_samples_inv_rej)
        self._update_noise_eps(init_eps=self.noise_eps)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                self.policy.turn_on_explore_mode() # Turn on exploration mode
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                (actions, values, log_probs, actions_val_all, num_val_samples, num_samples) \
                    = self.sample_act_fixed(obs_tensor, self.n_samples_inv_rej)
                self.policy.turn_off_explore_mode() # Turn off exploration mode
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value
            rollout_buffer.add(
                self._last_obs, actions, rewards, self._last_episode_starts, values, 
                log_probs, actions_val_all, num_val_samples, num_samples)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def sample_act_fixed(
        self, 
        obs: th.Tensor, 
        n_samples: int, 
        n_samples_max: int = 128,
        n_iter_max: int = 20
    ):
        """
        Sample fixed number of actions per obs, rejecting invalid actions

        Note: Flow policy has no deterministic mode
        """
        no_valid_samples = True # No valid action found in a certain state
        n_iter = 0
        while no_valid_samples:
            # Extend obs
            # obs_n_repeat = repeat(obs, "b ... -> (b r) ...", r=n_samples)
            obs_n_repeat = op_on_obs(repeat, obs, pattern="b ... -> (b r) ...", r=n_samples) 

            # Sample actions
            with th.no_grad(): 
                features = self.policy.extract_features(obs_n_repeat)
                latent_pi, latent_vf = self.policy.mlp_extractor(features)
                values = self.policy.value_net(latent_vf)
                actions = self.policy.flow_net.sample(latent_pi).squeeze(1)

                # Dummy log_probs
                log_probs = th.zeros(size=(actions.size(0),), device=self.device)

            ## Validate actions
            if self.enable_parallel:
                val_idx = np.array(
                    self.parallel(delayed(self.act_check_fn)(a,o) for a, o 
                    in zip(actions.data.cpu().numpy(), obs_n_repeat.data.cpu().numpy()))
                )
            else:
                if isinstance(obs_n_repeat, dict):
                    actions_ = actions.data.cpu().numpy()
                    obs_n_repeat_ = OrderedDict([(k, v.data.cpu().numpy()) for k, v in obs_n_repeat.items()])
                    val_idx = np.array([
                        self.act_check_fn(x[0], {k: v for k, v in zip(obs_n_repeat_.keys(), x[1:])}) 
                        for x in zip(actions_, *obs_n_repeat_.values())
                    ])
                else:
                    val_idx = np.array([
                        self.act_check_fn(a, o) for a, o 
                        in zip(actions.data.cpu().numpy(), obs_n_repeat.data.cpu().numpy())
                    ])
            ## At least one valid action?
            num_val_samples = rearrange(val_idx, "(b r) -> b r", r=n_samples).sum(1).astype(int)
            if not np.all(num_val_samples>0):
                # No valid action found in a certain state
                n_samples = min(n_samples_max, 2*n_samples)
            else:
                no_valid_samples = False
            
            if n_iter > n_iter_max:
                # note: random sample if reach max iteration and continue
                # raise RuntimeError("No valid action found in a certain state")

                # random sample
                for _obs_idx in np.where(num_val_samples==0)[0]:
                    _random_idx_of_obs = np.random.choice(np.arange(_obs_idx*n_samples, (_obs_idx+1)*n_samples))
                    val_idx[_random_idx_of_obs] = True
                no_valid_samples = False
            n_iter += 1
            
        # Sample from valid actions
        val_sample_idx = np.apply_along_axis(
            lambda x: np.random.choice(np.where(x)[0]), 1, 
            rearrange(val_idx, "(b r) -> b r", r=n_samples)
        )
        val_sample_idx = val_sample_idx + np.arange(val_idx.shape[0], step=n_samples)
        acts_val, values, log_probs = (
            actions[val_sample_idx], values[val_sample_idx], log_probs[val_sample_idx])
        
        # Valid action, num of samples
        acts_val_all = actions[val_idx].data.cpu().numpy()
        num_samples = np.full(num_val_samples.shape, n_samples).astype(int)

        return (acts_val, values, log_probs, 
                acts_val_all, num_val_samples, num_samples)

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        with th.no_grad():
            obs_tensor = obs_as_tensor(observation, self.device)
            action = self.sample_act_fixed(obs_tensor, self.n_samples_inv_rej)[0]
            action = action.cpu().numpy()
        assert state is None
        return action, state

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # Use the Rollout buffer that saves all valid action samples
        observation_space = self.observation_space
        buffer_cls = ObjActDictRolloutBuffer if isinstance(observation_space, spaces.Dict) else ObjActRolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def _update_sample_size(self, n_max=64, n_min=8, alpha=0.8, update_interval=10000):
        """
        Update the samples size used for invalid action rejection sampling
        """
        if self.num_timesteps - self._last_update_sample_size_timestep > update_interval:
            self._last_update_sample_size_timestep = self.num_timesteps
            self.n_samples_inv_rej = max(n_min, min(n_max, int(n_min * alpha/self._valid_sample_rate)))

    def _process_action_samples(self, rollout_data):
        actions_val_all = rollout_data.actions_val_all
        num_val_samples = rollout_data.num_val_samples
        num_samples = rollout_data.num_samples
        obs = rollout_data.observations

        # Convert numpy arr to tensor
        actions_val_all = np.concatenate(actions_val_all, axis=0)
        actions_val_all = obs_as_tensor(actions_val_all, self.device)
        num_val_samples = obs_as_tensor(num_val_samples, self.device)
        num_samples = obs_as_tensor(num_samples, self.device)

        # Compute sum of action log probs
        if isinstance(obs, dict):
            obs_ext = {k: v.repeat_interleave(num_val_samples.long(), dim=0) for k, v in obs.items()}
        else:
            obs_ext = obs.repeat_interleave(num_val_samples.long(), dim=0)
        _, log_prob_val_all, __ = self.policy.evaluate_actions(obs_ext, actions_val_all)
        log_prob_val_sum = th.zeros((num_samples.size(0),), device=self.device, dtype=th.float32)
        idx_sum = th.repeat_interleave(
                th.arange(num_samples.size(0)).to(self.device), num_val_samples.long(), dim=0)
        log_prob_val_sum = th.index_add(log_prob_val_sum, 0, idx_sum, log_prob_val_all)

        assert num_val_samples.requires_grad == False
        assert num_samples.requires_grad == False

        return log_prob_val_sum, num_val_samples, num_samples
    
    def _update_noise_eps(
        self, 
        init_eps=0, 
        final_eps=None, 
        ratio_to_final_eps=0.5
    ): 
        """
        Args:
            ratio_to_final_eps: proportion of the training steps that the eps 
                decreases by the final noise eps
        """
        cpr = self._current_progress_remaining # from 1 to 0
        rtf = ratio_to_final_eps
        ie = init_eps
        fe = final_eps

        if ie == 0.0:
            return self.policy.set_noise_eps(0.0)
        
        if fe is None:
            fe = 0.01

        if cpr > rtf:
            # Linearly decrease noise eps
            new_eps = ie + (cpr-1)*(fe-ie)/(rtf-1)
            self.policy.set_noise_eps(new_eps)
        else:
            # keep noise eps constant
            new_eps = fe
            self.policy.set_noise_eps(new_eps)

        self.logger.record("train/noise_eps", new_eps)
        
    def _reg_loss_w_base(self, obs):
        """
        Compute regularizer loss between the reference N(0,1) and the flow 
            distribution p(v) (marginalize over s).
        """
        if self.reg_coef_1 == 0.0:
            return th.tensor([0.0], device=self.device)
        rep = self.policy.rep_reg
        obs_ext = op_on_obs(repeat, obs, pattern="b ... -> (b r) ...", r=rep)
        return self.policy.reg_ref_dist(obs_ext)

def op_on_obs(fn: Callable, obs, **kwargs):
    """
    Apply a function on the observation.
    """
    if isinstance(obs, dict):
        return OrderedDict([(k, fn(v, **kwargs)) for k, v in obs.items()])
    elif isinstance(obs, np.ndarray) or isinstance(obs, th.Tensor):
        return fn(obs, **kwargs)