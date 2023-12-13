from collections import defaultdict
from copy import deepcopy
from typing import List

import cv2
import numpy as np
from gym import spaces
from pettingzoo.butterfly import pistonball_v4

try: 
    from ..util import ParallelToGymWrapper
except:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
    from envs.util import ParallelToGymWrapper

class PistonToGymWrapper(ParallelToGymWrapper): 
    def __init__(
        self,
        obs_type: str='global',
        continuous: bool=False,
        **kwargs,
    ):
        super().__init__(pistonball_v4.parallel_env(continuous=continuous, **kwargs))

        self._check_aec_spaces(self.aec_env)

        self.action_space = spaces.MultiDiscrete([
            actspa.n for actspa in self.aec_env.action_spaces.values()
        ])
        if obs_type == "global": 
            sample_obs = self.global_obs
            self.observation_space = spaces.Box(
                low=0, high=255, shape=sample_obs.shape, dtype=np.uint8
            )
        elif obs_type == "global_processed":
            # process the observation by grayscale, resize, and stack
            self.stack_size = 1
            self.resize_shape = (96, 96)
            self.observation_space = spaces.Box(
                low=0, high=255, shape=self.resize_shape+(self.stack_size,), dtype=np.uint8
            )
        else: 
            raise NotImplementedError
        
        self.obs_type = obs_type

        if type(self) is PistonToGymWrapper: self.reset()
    
    def step(self, action: np.ndarray):
        assert action.ndim == 1
        assert action.dtype in [np.int32, np.int64]

        rewards = defaultdict(int)
        actions = self.unbatchify(action)

        # =================== DO NOT CHANGE ===================
        for agent in self.aec_env.agents:
            if agent != self.aec_env.agent_selection:
                if self.aec_env.dones[agent]:
                    raise AssertionError(f"expected agent {agent} got done agent {self.aec_env.agent_selection}. Parallel environment wrapper expects all agent termination (setting an agent's self.dones entry to True) to happen only at the end of a cycle.")
                else:
                    raise AssertionError(f"expected agent {agent} got agent {self.aec_env.agent_selection}, Parallel environment wrapper expects agents to step in a cycle.")
            obs, rew, done, info = self.aec_env.last()
            self.aec_env.step(actions[agent])
            for agent in self.aec_env.agents:
                rewards[agent] += self.aec_env.rewards[agent]

        dones = dict(**self.aec_env.dones)
        infos = dict(**self.aec_env.infos)
        # =================== DO NOT CHANGE ===================

        if self.obs_type == "global": 
            observation = self.global_obs
        elif self.obs_type == "global_processed":
            observation = self.global_obs_processed
        else:
            raise NotImplementedError
        
        reward = np.mean(list(rewards.values())).item() # Mean over all agents' rewards
        done = np.all(list(dones.values())).item() # True if all agents are done
        info = infos
        info.update({"nearby_pistons": self.aec_env.get_nearby_pistons()})

        while self.aec_env.agents and self.aec_env.dones[self.aec_env.agent_selection]:
            self.aec_env.step(None)

        self.agents = self.aec_env.agents
        return observation, reward, done, info
    
    def reset(self):
        self.p_env.reset()

        if self.obs_type == "global": 
            observation = self.global_obs
        elif self.obs_type == "global_processed":
            observation = self.global_obs_processed
        else:
            raise NotImplementedError
        
        self.agents = deepcopy(self.aec_env.agents)
        return observation

    def _check_aec_spaces(self, aec_env):
        # observation_spaces
        obs_space_agent = list(aec_env.observation_spaces.values())[0]
        obs_low, obs_high = obs_space_agent.low, obs_space_agent.high
        assert np.all(obs_low == obs_low.max()), "observation space should have the same lower bound"
        assert np.all(obs_high == obs_high.max()), "observation space should have the same upper bound"

        # action_spaces
        nvec = np.array([actspa.n for actspa in aec_env.action_spaces.values()])
        ## only consider the case where all agents have the same action space
        assert np.all(nvec == nvec[0]), "each agent should have the same action space"
    
    def state(self):
        return self.aec_env.state()
    
    @property
    def global_obs(self): 
        return self.state().astype(np.uint8)
    
    @property
    def global_obs_processed(self):
        obs = self.global_obs
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(
            obs, self.resize_shape[::-1], interpolation=cv2.INTER_AREA)
        obs = np.expand_dims(obs, axis=-1)
        
        return obs
        
    
class PistonWithCstrToGymWrapper(PistonToGymWrapper):
    def __init__(
        self, 
        idx_pistons_cstr: np.ndarray=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.observation_space = spaces.Dict({
            "observation": self.observation_space,
            "status_for_cstr_oracle": spaces.Discrete(self.aec_env.n_pistons)
        })

        if idx_pistons_cstr is None:
            n_pistons = self.aec_env.n_pistons
            self.idx_pistons_cstr = np.arange(n_pistons)
        else:
            self.idx_pistons_cstr = idx_pistons_cstr

        if type(self) is PistonWithCstrToGymWrapper: self.reset()

    def step(self, action: np.ndarray):
        # check action validity 
        validity = self.act_check(action, {"status_for_cstr_oracle": 
                                   min(self.aec_env.get_nearby_pistons())})

        # map the action into the feasible region
        safe_action = action
        if not validity:
            # conservative: set the action to 0
            safe_action = np.zeros_like(action, dtype=action.dtype)

        # perform the action
        observation, reward, done, info = super().step(safe_action)
        observation = {
            "observation": observation,
            "status_for_cstr_oracle": min(info['nearby_pistons'])
        }

        # extra info
        info.update({"valid": validity})

        return observation, reward, done, info
    
    def reset(self):
        observation = super().reset()
        observation = {
            "observation": observation,
            "status_for_cstr_oracle": min(self.aec_env.get_nearby_pistons())
        }
        return observation
    
    def act_check(
        self, 
        action: np.ndarray,
        observation,
    ) -> bool:
        """Check if the action is valid depending on the observation. 
        
        Note:
            This function is independent of the environment state.
        """
        min_idx_near = observation["status_for_cstr_oracle"]
        return self._no_moveup_check(action, min_idx_near)
    
    def gen_mask_from_obs(self, obs, cand_act: np.ndarray) -> np.ndarray:
        """
        Generate mask from observation and candidate actions.
        
        Returns:
            masks: (num_act,) array, 1 if the action is valid, 0 otherwise.
        """
        assert cand_act.ndim == 2, "cand_act should be 2D array"
        return np.apply_along_axis(self.act_check, 1, cand_act, obs)
    
    def _no_moveup_check(
        self,
        action: np.ndarray,
        min_idx_near: int,
    ) -> bool:
        """Valid if the pistons on the left side of the ball do not move up."""
        idx_pc = self.idx_pistons_cstr
        idx_pistons_left = np.array([i for i in idx_pc if i < min_idx_near], dtype=np.int32)
        return np.all(action[idx_pistons_left] != 2)

if __name__ == "__main__":
    import sys
    import warnings

    from stable_baselines3.common.env_checker import check_env

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    def done_check(env, max_cycle=500):
        done = False
        i = 0
        
        while not done: 
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            i += 1

        if i in [max_cycle, max_cycle+1, max_cycle-1]:
            print("done check passed")
        else:
            print("done check not passed")

    def valid_check(env):
        _act = env.action_space.sample()
        valid_action = np.zeros(_act.shape, dtype=_act.dtype)
        invalid_action = np.ones(_act.shape, dtype=_act.dtype) * 2

        validity = []
        for _ in range(20):
            obs, reward, done, info = env.step(invalid_action)
            validity.append(info["valid"])
        for _ in range(20):
            obs, reward, done, info = env.step(valid_action)
            validity.append(info["valid"])
        
        if np.all(validity[20:]) and not np.any(validity[:20]):
            print("valid check passed")
        else:
            print("valid check not passed")

    # ========== Test Pistonball w/o constraints ==========
    print("\n========== Test Pistonball w/o constraints ==========")
    # piston env
    env = PistonToGymWrapper(n_pistons=20)
    check_env(env)
    env.reset()
    done_check(env, 125)


    # ========== Test Pistonball w/ constraints ==========
    print("\n========== Test Pistonball w/ constraints ==========")
    env = PistonWithCstrToGymWrapper(n_pistons=20)
    check_env(env)
    env.reset()
    done_check(env, 125)
    env.reset()
    valid_check(env)