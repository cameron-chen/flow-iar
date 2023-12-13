from collections import defaultdict
from copy import deepcopy

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
        else: 
            raise NotImplementedError
        
        self.obs_type = obs_type
    
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
        else:
            raise NotImplementedError
        
        reward = np.mean(list(rewards.values())).item() # Mean over all agents' rewards
        done = np.all(list(dones.values())).item() # True if all agents are done
        info = infos

        while self.aec_env.agents and self.aec_env.dones[self.aec_env.agent_selection]:
            self.aec_env.step(None)

        self.agents = self.aec_env.agents
        return observation, reward, done, info
    
    def reset(self):
        self.p_env.reset()

        if self.obs_type == "global": 
            observation = self.global_obs
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
    
if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

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

    # piston env
    env = PistonToGymWrapper(n_pistons=8)
    check_env(env)
    done_check(env, 114)