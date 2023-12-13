from typing import Dict, Generator, NamedTuple, Optional, Union

import numpy as np
import torch as th
from gym import spaces
from scipy.stats import wasserstein_distance
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.preprocessing import (get_action_dim,
                                                    get_obs_shape)
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.vec_env import VecNormalize

from utils.flows.CategoricalNF.networks.autoregressive_layers import \
    AutoregressiveLSTMModel


def stat_flow_act_and_log_prob(
    acts: np.ndarray,
    log_prob: np.ndarray,
)-> np.ndarray:
    """Compute the statistics of flow acts and log_prob.

    Returns:
        act_stat: ndarray, columns - (act, mean_prob, std_prob, emp_prob)
        emd: wasserstein distance between the flow distribution and the empirical distribution
    """
    n_acts = acts.max() + 1
    act_stat = np.zeros((n_acts, 4))
    act_stat[:, 0] = np.arange(n_acts)

    for i in range(n_acts):
        idx = np.where(acts == i)[0]
        act_stat[i, 1] = np.mean(np.exp(log_prob[idx])) if len(idx) > 0 else 0.0
        act_stat[i, 2] = np.std(np.exp(log_prob[idx])) if len(idx) > 0 else 0.0
        act_stat[i, 3] = len(idx)/len(log_prob)
    
    emd = wasserstein_distance(act_stat[:, 0], act_stat[:, 0], 
                               u_weights=act_stat[:,1], v_weights=act_stat[:, 3])

    return act_stat, emd

def ar_func(c_in, c_out, hidden, num_layers, max_seq_len, input_dp_rate):
    return AutoregressiveLSTMModel(
        c_in=c_in,
        c_out=c_out,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        hidden_size=hidden,
        dp_rate=0,
        input_dp_rate=input_dp_rate)

class BufferSingleAttr(object):
    """Buffer for single attribute.

    Note: 
        The buffer ignores the number of environments.
    """
    def __init__(
        self, 
        attr_space: spaces.Space,
        buffer_size: int=2048,
        device: Union[th.device, str] = "cpu",
        attr_type: str = "obs", # "obs" or "act"
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.attr_space = attr_space
        self.device = device
        self.attr = None
        self.full = False

        if attr_type=="obs":self.attr_shape = get_obs_shape(attr_space)
        else:self.attr_shape = (get_action_dim(attr_space),)

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.attr is None: return 0
        return self.attr.shape[0]

    def add(self, attr: np.ndarray):
        assert attr.shape[1:] == self.attr_shape

        if self.attr is None: 
            self.attr = attr
        else:
            self.attr = np.concatenate((self.attr, attr), axis=0)

        self.attr = self.attr[-self.buffer_size:]

        if self.full: return
        elif self.size() == self.buffer_size:
            self.full = True

    def get(self)->np.ndarray:
        return self.attr

    def sample(self, sample_size: int)->np.ndarray:
        return self.attr[np.random.choice(self.size(), sample_size, replace=False)]
    
class DictBufferSingleAttr(object):
    """Buffer for single dictionary attribute.

    Note: 
        The buffer ignores the number of environments.
    """
    def __init__(
        self, 
        attr_space: spaces.Space,
        buffer_size: int=2048,
        device: Union[th.device, str] = "cpu",
    ):
        assert isinstance(attr_space, spaces.Dict)

        super().__init__()
        self.buffer_size = buffer_size
        self.attr_space = attr_space
        self.device = device
        self.attr = None
        self.full = False
        self.attr_shape = get_obs_shape(attr_space)

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.attr is None: 
            return 0
        for _, v in self.attr.items():
            return v.shape[0]

    def add(self, new_data: Dict[str, np.ndarray]):
        buffer_size = self.buffer_size
        attr_shape = self.attr_shape

        for k, v in new_data.items():
            assert v.shape[1:] == attr_shape[k]

        if self.attr is None: 
            self.attr = new_data
        else:
            for k, v in new_data.items():
                self.attr[k] = np.concatenate((self.attr[k], v), axis=0)

        self.attr = {k: v[-buffer_size:] for k, v in self.attr.items()}

        if self.full: return
        elif self.size() == buffer_size:
            self.full = True

    def get(self)->Dict[str, np.ndarray]:
        return self.attr

    def sample(self, sample_size: int)->Dict[str, np.ndarray]:
        idx = np.random.choice(self.size(), sample_size, replace=False)
        return {k: v[idx] for k, v in self.attr.items()}

class ActRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    actions_val_all: np.ndarray
    num_val_samples: np.ndarray
    num_samples: np.ndarray
    
class ActDictRolloutBufferSamples(ActRolloutBufferSamples):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    actions_val_all: np.ndarray
    num_val_samples: np.ndarray
    num_samples: np.ndarray

class ObjActRolloutBuffer(RolloutBuffer): 
    """
    Object Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RolloutBuffer to save all the sampled valid actions in an array of object type. 

    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)
        self.actions_val_all, self.num_val_samples, self.num_samples = None, None, None
        self.reset()
    
    def reset(self) -> None:
        self.actions_val_all = np.zeros((self.buffer_size, self.n_envs), dtype=object)
        self.num_val_samples = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.num_samples = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        super().reset()

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        actions_val_all: np.ndarray,
        num_val_samples: np.ndarray,
        num_samples: np.ndarray,
    ) -> None:
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.num_val_samples[self.pos] = np.array(num_val_samples).copy()
        self.num_samples[self.pos] = np.array(num_samples).copy()

        start_idx = 0
        for i, n in enumerate(num_val_samples):
            self.actions_val_all[self.pos, i] = np.array(actions_val_all[start_idx:start_idx+n]).copy()
            start_idx += n

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[ActRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "actions_val_all",
                "num_val_samples",
                "num_samples",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])

            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ActRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )

        data_arr = (
            self.actions_val_all[batch_inds].flatten(),
            self.num_val_samples[batch_inds].flatten(),
            self.num_samples[batch_inds].flatten(),
        )
        return ActRolloutBufferSamples(*tuple(map(self.to_torch, data)), *data_arr)
    
class ObjActDictRolloutBuffer(DictRolloutBuffer): 
    """
    Object Rollout buffer with dictionary observation used in on-policy algorithms like A2C/PPO.
    Extends the RolloutBuffer to save all the sampled valid actions in an array of object type. 

    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)
        self.actions_val_all, self.num_val_samples, self.num_samples = None, None, None
        self.reset()
    
    def reset(self) -> None:
        self.actions_val_all = np.zeros((self.buffer_size, self.n_envs), dtype=object)
        self.num_val_samples = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.num_samples = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        super().reset()

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        actions_val_all: np.ndarray,
        num_val_samples: np.ndarray,
        num_samples: np.ndarray,
    ) -> None:
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = np.array(obs[key]).copy()
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.num_val_samples[self.pos] = np.array(num_val_samples).copy()
        self.num_samples[self.pos] = np.array(num_samples).copy()

        start_idx = 0
        for i, n in enumerate(num_val_samples):
            self.actions_val_all[self.pos, i] = np.array(actions_val_all[start_idx:start_idx+n]).copy()
            start_idx += n

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[ActRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = [
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "actions_val_all",
                "num_val_samples",
                "num_samples",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])

            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ActDictRolloutBufferSamples:
        
        return ActDictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            actions_val_all=self.actions_val_all[batch_inds].flatten(),
            num_val_samples=self.num_val_samples[batch_inds].flatten(),
            num_samples=self.num_samples[batch_inds].flatten(),
        )