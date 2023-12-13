from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from einops import rearrange, repeat
from gym import spaces
from joblib import Parallel
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import (GymEnv, MaybeCallback,
                                                   RolloutReturn, Schedule,
                                                   TrainFreq,
                                                   TrainFrequencyUnit)
from stable_baselines3.common.utils import (obs_as_tensor,
                                            should_collect_more_steps)
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.ddpg.ddpg import DDPG
from stable_baselines3.td3.policies import TD3Policy

from .action_space import Discrete_space, Multi_discrete_space, Space


class Wol_DDPG(DDPG):
    """Wolpertinger training with DDPG. 
    """
    def __init__(
        self, 
        policy: Union[str, Type[TD3Policy]], 
        env: Union[GymEnv, str], 
        learning_rate: Union[float, Schedule] = 0.001, 
        buffer_size: int = 1000000, 
        learning_starts: int = 100, 
        batch_size: int = 100, 
        tau: float = 0.005, 
        gamma: float = 0.99, 
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),   #Check
        gradient_steps: int = -1, 
        action_noise: Optional[ActionNoise] = None, 
        replay_buffer_class: Optional[ReplayBuffer] = None, 
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None, 
        optimize_memory_usage: bool = False, 
        tensorboard_log: Optional[str] = None, 
        create_eval_env: bool = False, 
        policy_kwargs: Optional[Dict[str, Any]] = None, 
        verbose: int = 0, seed: Optional[int] = None, 
        device: Union[th.device, str] = "auto", 
        _init_setup_model: bool = True,
        k_ratio: float = 0.1,
        df_max_actions: int = 200000,
        mode: str = 'medium'
        ):
        self.continuous, self.multi_discrete, max_actions, action_low, action_high, nb_states, nb_actions = self.pre_init(env)

        if self.continuous:
            self.orig_action_space = Space(action_low, action_high, df_max_actions, mode=mode)
            self.k_nearest_neighbors = max(1, int(df_max_actions * k_ratio))
        else:
            if not self.multi_discrete: #1 dimensional discrete
                self.orig_action_space = Discrete_space(max_actions, mode=mode)
                self.k_nearest_neighbors = max(1, int(max_actions * k_ratio))
            else: #multi discrete action space
                self.orig_action_space = Multi_discrete_space(action_low, action_high, max_actions, mode=mode)
                self.k_nearest_neighbors = max(1, int(max_actions * k_ratio))

        env = self.convert_action_space(env)

        super().__init__(policy, env, learning_rate, buffer_size, learning_starts, batch_size, tau, gamma, train_freq, gradient_steps, action_noise, replay_buffer_class, replay_buffer_kwargs, optimize_memory_usage, tensorboard_log, create_eval_env, policy_kwargs, verbose, seed, device, _init_setup_model)
        
        if max_actions is not None:
            assert self.k_nearest_neighbors <= max_actions, \
                "k_nearest_neighbors must be less than or equal to max_actions"
        else: assert self.k_nearest_neighbors <= df_max_actions
        assert self.k_nearest_neighbors > 0, "k_nearest_neighbors must be greater than 0"

    def pre_init(self, env):
        """Identify characteristics of the action space before initializing DDPG

        """

        continuous = None
        multi_discrete = None
        max_actions= None 
        action_low= None 
        action_high= None 
        nb_states=None
        nb_actions= None
        if isinstance(env.action_space, spaces.Box):
            nb_states = env.observation_space.shape[0]
            nb_actions = env.action_space.shape[0]
            action_high = env.action_space.high
            action_low = env.action_space.low
            continuous = True
        elif isinstance(env.action_space, spaces.Discrete):     #discrete action for 1 dimension
            nb_states = env.observation_space.shape[0]
            nb_actions = 1  # the dimension of actions
            max_actions = env.action_space.n
            continuous = False
            multi_discrete = False
        elif isinstance(env.action_space, spaces.MultiDiscrete):
            nb_states = None # env.observation_space.shape[0]
                             # Hack: return None for nb_states as not used while cause 
                             #      issue with dict observation space
            nb_actions = env.action_space.shape[0]
            continuous = False
            multi_discrete = True
            action_high = np.array(env.action_space.nvec) - 1
            action_low = np.zeros_like(env.action_space.nvec)
            max_actions = np.prod(env.action_space.nvec)
        else:
            raise ValueError('gym_space not recognized')

        return continuous, multi_discrete, max_actions, action_low, action_high, nb_states, nb_actions


    def convert_action_space(self, env):
        """Convert action space to Box space to feed to DDPG

        """
        act_space = env.action_space

        if isinstance(act_space, spaces.Box):
            new_act_space = act_space
        elif isinstance(act_space, spaces.Discrete):     #discrete action for 1 dimension
            new_act_space = spaces.Box(low=np.float32(0), high= np.float32(act_space.n), 
                shape=act_space.shape, dtype=np.float32)
        elif isinstance(act_space, spaces.MultiDiscrete):
            new_act_space = spaces.Box(low=np.zeros_like(act_space.nvec).astype(np.float32), 
                high=np.array(act_space.nvec -1).astype(np.float32), dtype=np.float32)
        else:
            raise ValueError('gym_space not recognized')

        env.action_space = new_act_space
        return env
              
    # NOTE: Seems not necessary to override this method
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        return super().train(gradient_steps, batch_size)
    
    # NOTE: Seems not necessary to override this method
    def learn(self, total_timesteps: int, callback: MaybeCallback = None, log_interval: int = 4, eval_env: Optional[GymEnv] = None, eval_freq: int = -1, n_eval_episodes: int = 5, tb_log_name: str = "DDPG", eval_log_path: Optional[str] = None, reset_num_timesteps: bool = True) -> OffPolicyAlgorithm:
        return super().learn(total_timesteps, callback, log_interval, eval_env, eval_freq, n_eval_episodes, tb_log_name, eval_log_path, reset_num_timesteps)

    def collect_rollouts(
        self, 
        env: VecEnv, 
        callback: BaseCallback, 
        train_freq: TrainFreq, 
        replay_buffer: ReplayBuffer, 
        action_noise: Optional[ActionNoise] = None, 
        learning_starts: int = 0, 
        log_interval: Optional[int] = None) -> RolloutReturn:

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            proto_actions, _ = self._sample_action(learning_starts, action_noise, env.num_envs)

            if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
                # random
                ## buffer_actions (n_envs, fea): scaled action, [-1, 1]
                ## actions (n_envs, fea): unscaled action, [low, high]
                buffer_actions_k, actions_k = self.orig_action_space.search_point(proto_actions, 1)
                buffer_actions = buffer_actions_k[:, 0]
                actions = actions_k[:, 0]
            else:
                # according to policy
                ## buffer_actions (n_envs, fea): scaled action, [-1, 1]
                ## actions (n_envs, fea): unscaled action, [low, high]
                buffer_actions, actions = self.wolp_action(self._last_obs, proto_actions)

            # feed discrete actions to env
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # feed continuous actions to ddpg buffer
            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

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
        proto_actions, state = super().predict(observation, state, episode_start, deterministic)
        return self.wolp_action(observation, proto_actions)[-1], state

    def wolp_action(
        self, 
        obs: Union[np.ndarray, dict], 
        proto_action: np.ndarray,
    ):
        """Implement Wolpertinger algo and get actions"""
        assert isinstance(obs, np.ndarray) or isinstance(obs, dict)
        assert isinstance(proto_action, np.ndarray)

        # get the proto_action's k nearest neighbors
        ## raw_actions (n_envs, k, fea): scaled action, [-1, 1]
        ## actions (n_envs, k, fea): unscaled action, [low, high]
        raw_actions, actions = self.orig_action_space.search_point(proto_action, self.k_nearest_neighbors)

        # if k = 1, return the only action, 
        # otherwise, use the critic to choose the best action
        if self.k_nearest_neighbors == 1:
            return raw_actions[:, 0], actions[:, 0]

        # make all the state, action pairs for the critic
        raw_actions_tensor = obs_as_tensor(rearrange(raw_actions, 'n k ... -> (n k) ...'), self.device)
        obs_tensor = obs_as_tensor(
            op_on_obs(repeat, obs, pattern='n ... -> (n k) ...', k=self.k_nearest_neighbors),
            # repeat(obs, 'n ... -> (n k) ...', k=self.k_nearest_neighbors), 
            self.device)

        # evaluate each pair through the critic
        actions_evaluation = self.critic(obs_tensor, raw_actions_tensor)[0].data.cpu().numpy()
        # HACK: we assume only one q-network in the critic
        #       think about how to deal with multiple q-networks, e.g., TD3

        # find the index of the pair with the maximum value
        ## max_index (n_envs,): the index of the action with the maximum value
        max_index = np.argmax(
            rearrange(actions_evaluation, '(n k) 1 -> n k', k=self.k_nearest_neighbors), axis=1)

        # return the best action, i.e., wolpertinger action from the full wolpertinger policy
        return raw_actions[np.arange(proto_action.shape[0]), max_index], actions[np.arange(proto_action.shape[0]), max_index]
        
class Wol_DDPG_Rej(Wol_DDPG):
    """Wolpertinger DDPG with invalid action rejection. 
    """
    def __init__(
        self, 
        policy: Union[str, Type[TD3Policy]], 
        env: Union[GymEnv, str], 
        learning_rate: Union[float, Schedule] = 0.001, 
        buffer_size: int = 1000000, 
        learning_starts: int = 100, 
        batch_size: int = 100, 
        tau: float = 0.005, 
        gamma: float = 0.99, 
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1, 
        action_noise: Optional[ActionNoise] = None, 
        replay_buffer_class: Optional[ReplayBuffer] = None, 
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None, 
        optimize_memory_usage: bool = False, 
        tensorboard_log: Optional[str] = None, 
        create_eval_env: bool = False, 
        policy_kwargs: Optional[Dict[str, Any]] = None, 
        verbose: int = 0, seed: Optional[int] = None, 
        device: Union[th.device, str] = "auto", 
        _init_setup_model: bool = True,
        k_ratio: float = 0.1,
        df_max_actions: int = 60000,
        n_jobs: int = -1,
        mode: str = 'medium'
    ):
        """
        Args:
            df_max_actions: upper bound of the number of actions.
                Considering the time cost on validate the actions, the recommended value 
                is 60000.
        """
        # FIXME: need to overwrite the predict function as it outputs the original continuous 
        # action of DDPG (proto_action). 

        super().__init__(policy, env, learning_rate, buffer_size, 
            learning_starts, batch_size, tau, gamma, train_freq, gradient_steps, 
            action_noise, replay_buffer_class, replay_buffer_kwargs, optimize_memory_usage, 
            tensorboard_log, create_eval_env, policy_kwargs, verbose, seed, device, 
            _init_setup_model, k_ratio, df_max_actions, mode=mode)
        
        self.k_nearest_neighbors = min(self.k_nearest_neighbors, int(df_max_actions * k_ratio))

        try: 
            self.act_check_fn = env.envs[0].act_check
        except:
            raise RuntimeError("The environment does not have act_check function.")
        
        self.n_val_acts = []
        self.enable_parallel = True if n_jobs > 1 else False
        if self.enable_parallel: self.workers = Parallel(n_jobs=n_jobs)
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """Log valid action rate in the training function."""
        super().train(gradient_steps, batch_size)
        
        if self.n_val_acts:
            self.logger.record("train/val_rate_sample", np.mean(self.n_val_acts)/self.k_nearest_neighbors)
            self.n_val_acts = []
    
    def wolp_action(
        self, 
        obs: Union[np.ndarray, dict], 
        proto_action: np.ndarray,
    ):
        """
        Wolpertinger algo with invalid action rejection, i.e. reject invalid
            actions and choose the one with the maximal value in valid actions.
            
            Considering fault tolerance, we allow outputing invalid actions.
        
        Returns:
            raw_actions (n_envs, fea): scaled action, [-1, 1]
            actions (n_envs, fea): unscaled action, [low, high]
        """
        assert isinstance(obs, np.ndarray) or isinstance(obs, dict)
        assert isinstance(proto_action, np.ndarray)

        # get the proto_action's k nearest neighbors
        raw_actions, actions = self.orig_action_space.search_point(proto_action, self.k_nearest_neighbors)

        # if k = 1, return the only action, not check validity in this case.
        # otherwise, use the critic to choose the best action
        if self.k_nearest_neighbors == 1:
            return raw_actions[:, 0], actions[:, 0]
        
        # check the validity of the actions
        validity, n_val_act = self.check_validity(obs, actions)

        # if all actions given an obs are invalid, evaluate all actions
        # else, evaluate the valid actions
        selected_bool = validity.copy()
        selected_num = n_val_act.copy()
        for idx in np.where(n_val_act == 0)[0]:
            # if all actions are invalid, select all actions to evaluate
            selected_bool[idx] = np.ones(self.k_nearest_neighbors, dtype=bool)
            selected_num[idx] = self.k_nearest_neighbors

        # selected_idx has index in the range (0, n_envs * k)
        selected_idx = np.where(rearrange(selected_bool, "n k -> (n k)"))[0]

        # make all the state, action pairs for the critic
        raw_actions_tensor = obs_as_tensor(
            rearrange(raw_actions, 'n k ... -> (n k) ...')[selected_idx], 
            self.device)
        obs_tensor = obs_as_tensor(
            op_on_obs(np.repeat, obs, repeats=selected_num, axis=0),
            # np.repeat(obs, selected_num, axis=0), 
            self.device)

        # evaluate each pair through the critic
        actions_evaluation = self.critic(obs_tensor, raw_actions_tensor)[0].data.cpu().numpy()
        # HACK: we assume only one q-network in the critic
        #       think about how to deal with multiple q-networks, e.g., TD3

        # find the index of the pair with the maximum value
        max_index = []
        start_idx = 0
        for i, n in enumerate(selected_num):
            _pos_max = np.argmax(actions_evaluation[start_idx:start_idx + n])
            pos_max = _pos_max + start_idx
            max_index.append(selected_idx[pos_max])
            start_idx += n
        max_index = np.array(max_index)

        # save data for logging
        self.n_val_acts.append(n_val_act)

        # return the best action, i.e., wolpertinger action from the full wolpertinger policy
        return rearrange(raw_actions, 'n k ... -> (n k) ...')[max_index], rearrange(actions, 'n k ... -> (n k) ...')[max_index]
    
    def check_validity(
        self, 
        obs: Union[np.ndarray, dict], 
        actions: np.ndarray
    )-> Tuple[np.ndarray, np.ndarray]:
        """Check the validity of the actions.
        
        Args: 
            obs (n_envs, obs_dim): the observation, obs_dim can be n-D
            actions (n_envs, k, act_dim): the actions, act_dim can be 0D or 1D

        Returns: 
            validity (n_envs, k): an array of bool, True if the action is valid
            n_val_act (n_envs,): the number of valid actions
        """
        n_envs = actions.shape[0]
        obs_ext = op_on_obs(repeat, obs, pattern="n ... -> (n k) ...", k=self.k_nearest_neighbors)
        # repeat(obs, "n ... -> (n k) ...", k=self.k_nearest_neighbors)
        actions = rearrange(actions, "n k ... -> (n k) ...")
        if not self.enable_parallel:
            if isinstance(obs, dict):
                validity = np.array([
                    self.act_check_fn(x[0], {k: v for k, v in zip(obs_ext.keys(), x[1:])}) 
                    for x in zip(actions, *obs_ext.values())
                ])
            else:
                validity = np.array([self.act_check_fn(a, o) for a, o in zip(actions, obs_ext)])
        else:
            raise NotImplementedError("Parallelization is not implemented yet.")
            # ToDo: support dict observations
            # validity = parallelize_loop(self.act_check_fn, self.workers, (actions, obs_ext))

        validity = rearrange(validity, "(n k) -> n k", n=n_envs)
        n_val_act = np.sum(validity, axis=1)

        return validity, n_val_act

def op_on_obs(fn: Callable, obs, **kwargs):
    """
    Apply a function on the observation.

    Note: place it here to avoid a circular import.
    """
    if isinstance(obs, dict):
        return OrderedDict([(k, fn(v, **kwargs)) for k, v in obs.items()])
    elif isinstance(obs, np.ndarray) or isinstance(obs, th.Tensor):
        return fn(obs, **kwargs)