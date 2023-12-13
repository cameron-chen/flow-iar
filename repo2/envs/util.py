import importlib
import os
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

import gym
import numpy as np
from gym.wrappers import GrayScaleObservation, ResizeObservation, TimeLimit
from pettingzoo.utils.conversions import to_parallel_wrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecEnv, VecEnvWrapper,
                                              VecFrameStack, VecNormalize)


class SubprocVecEnv_v2(SubprocVecEnv):
    """
    Child class of SubprocVecEnv that allows to add extra attributes.
    """
    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None):
        super().__init__(env_fns, start_method)
        self.envs = [env_fns[0]()]

class ActMapVecEnvWrapper(VecEnvWrapper):
    """Apply a mapping function to the actions.
    """
    def __init__(
        self, 
        venv: VecEnv, 
        fn: Callable,
        **fn_kwargs,
    ):
        assert not isinstance(venv, ActMapVecEnvWrapper)
        super().__init__(venv)
        self.fn = fn
        self.fn_kwargs = fn_kwargs

        self.reset()

    @property
    def envs(self):
        return self.venv.envs
    
    def reset(self):
        self._old_obs = self.venv.reset()
        return self._old_obs

    def step_async(self, actions: np.ndarray):
        actions = self.fn(actions, **self.fn_kwargs)
        self._actions = actions
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self._old_obs = obs
        return obs, rews, dones, infos

class ParallelToGymWrapper(gym.Env):
    def __init__(
        self, 
        parallel_env: to_parallel_wrapper,
    ):
        super().__init__()
        self.p_env = parallel_env
        self.aec_env = parallel_env.unwrapped
        self.agent_name_mapping = self.aec_env.agent_name_mapping
        self.action_space = None
        self.observation_space = None
        self.metadata = parallel_env.metadata
    
    def render(self, mode="human"):
        return self.p_env.render(mode)
    
    def close (self):
        return self.p_env.close()
    
    def seed(self, seed=None):
        return self.p_env.seed(seed)
    
    def unbatchify(self, arr: np.ndarray):
        """Converts np array to PZ style arguments."""
        return OrderedDict((a, arr[i]) for i, a in enumerate(self.p_env.possible_agents))

def assginment_rsc_to_stat_1d(assign, *args, **kwargs):
    """Convert an 1d assignment of resource-base form to a station-base form. 
    """
    return np.histogram(assign, *args, **kwargs)[0]

def assginment_rsc_to_stat_2d(assign, n_stations):
    """Convert a 2d assignment of resource form to a station form. 
    """
    return np.apply_along_axis(assginment_rsc_to_stat_1d, 1, assign, 
                               bins = np.arange(n_stations+1))


def get_wrapper_class(hyperparams: Dict[str, Any]) -> Optional[Callable[[gym.Env], gym.Env]]:
    """
    Get one or more Gym environment wrapper class specified as a hyper parameter
    "env_wrapper".
    e.g.
    env_wrapper: gym_minigrid.wrappers.FlatObsWrapper

    for multiple, specify a list:

    env_wrapper:
        - utils.wrappers.PlotActionWrapper
        - utils.wrappers.TimeFeatureWrapper


    :param hyperparams:
    :return: maybe a callable to wrap the environment
        with one or multiple gym.Wrapper
    """

    def get_module_name(wrapper_name):
        return ".".join(wrapper_name.split(".")[:-1])

    def get_class_name(wrapper_name):
        return wrapper_name.split(".")[-1]

    if "env_wrapper" in hyperparams.keys():
        wrapper_name = hyperparams.get("env_wrapper")

        if wrapper_name is None:
            return None

        if not isinstance(wrapper_name, list):
            wrapper_names = [wrapper_name]
        else:
            wrapper_names = wrapper_name

        wrapper_classes = []
        wrapper_kwargs = []
        # Handle multiple wrappers
        for wrapper_name in wrapper_names:
            # Handle keyword arguments
            if isinstance(wrapper_name, dict):
                assert len(wrapper_name) == 1, (
                    "You have an error in the formatting "
                    f"of your YAML file near {wrapper_name}. "
                    "You should check the indentation."
                )
                wrapper_dict = wrapper_name
                wrapper_name = list(wrapper_dict.keys())[0]
                kwargs = wrapper_dict[wrapper_name]
            else:
                kwargs = {}
            wrapper_module = importlib.import_module(get_module_name(wrapper_name))
            wrapper_class = getattr(wrapper_module, get_class_name(wrapper_name))
            wrapper_classes.append(wrapper_class)
            wrapper_kwargs.append(kwargs)

        def wrap_env(env: gym.Env) -> gym.Env:
            """
            :param env:
            :return:
            """
            for wrapper_class, kwargs in zip(wrapper_classes, wrapper_kwargs):
                env = wrapper_class(env, **kwargs)
            return env

        return wrap_env
    else:
        return None

def create_test_env(
    env_id: str,
    n_envs: int = 1,
    stats_path: Optional[str] = None,
    seed: int = 0,
    log_dir: Optional[str] = None,
    should_render: bool = True,
    hyperparams: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_normalize_final: bool = True,
    load_attr: bool = False,
) -> VecEnv:
    """
    Create environment for testing a trained agent

    :param env_id:
    :param n_envs: number of processes
    :param stats_path: path to folder containing saved running averaged
    :param seed: Seed for random number generator
    :param log_dir: Where to log rewards
    :param should_render: For Pybullet env, display the GUI
    :param hyperparams: Additional hyperparams (ex: n_stack)
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_normalize_final: Whether to load the vecnormalize_final or vecnormalize
    :return:
    """
    # Avoid circular import
    from utils.exp_manager import ExperimentManager

    # Create the environment and wrap it if necessary
    hyperparams = {} if hyperparams is None else hyperparams

    if "env_wrapper" in hyperparams.keys():
        del hyperparams["env_wrapper"]

    vec_env_kwargs = {}
    vec_env_cls = DummyVecEnv
    if n_envs > 1 or (ExperimentManager.is_bullet(env_id) and should_render):
        # HACK: force SubprocVecEnv for Bullet env
        # as Pybullet envs does not follow gym.render() interface
        vec_env_cls = SubprocVecEnv
        # start_method = 'spawn' for thread safe

    # Wrapper for image-based observation
    # HACK: not consider Dict observation for now
    _dummy_env = gym.make(env_id, **env_kwargs)
    is_image_based_env = is_image_space(_dummy_env.observation_space)

    if is_image_based_env:
        muesli_obs_size = 96
        def image_env_wrapper(env):
            env = GrayScaleObservation(env, keep_dim=True)
            env = ResizeObservation(env, (muesli_obs_size, muesli_obs_size))
            return env

    env = make_vec_env(
        env_id,
        n_envs=n_envs,
        monitor_dir=log_dir,
        seed=seed,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
        wrapper_class=image_env_wrapper if is_image_based_env else None,
    )

    if hyperparams["max_episode_steps"] is not None:
        env = TimeLimit(env, max_episode_steps=hyperparams.max_episode_steps)
    if "ERSEnv" in hyperparams["env_id"] and hyperparams["convert_act"]: 
        env = ActMapVecEnvWrapper(env, assginment_rsc_to_stat_2d, 
                    n_stations = env.action_space.nvec[0])

    # (Optional) Load environment attributes for testing
    if load_attr: load_attr_for_testing(env)

    # Load saved stats for normalizing input and rewards
    # And optionally stack frames
    if stats_path is not None:
        if hyperparams["normalize"]:
            print("Loading running average")
            print(f"with params: {hyperparams['normalize_kwargs']}")
            if vec_normalize_final: path_ = os.path.join(stats_path, "vecnormalize_final.pkl")
            else: path_ = os.path.join(stats_path, "vecnormalize.pkl") # Load the vecnormalize by EvalCallback
            if os.path.exists(path_):
                env = VecNormalize.load(path_, env)
                # Deactivate training and reward normalization
                env.training = False
                env.norm_reward = False
            else:
                raise ValueError(f"VecNormalize stats {path_} not found")

        n_stack = hyperparams.get("frame_stack", 0) if not is_image_based_env else hyperparams.get("frame_stack", 4)
        if n_stack > 0:
            print(f"Stacking {n_stack} frames")
            env = VecFrameStack(env, n_stack)
    return env

def load_attr_for_testing(venv: VecEnv):
    """
    Load environment attributes for testing.
        For example, we can assign the initial state of the environment by this function. 
    """
    # retrieve the necessary information
    try: 
        attrs = venv.envs[0].get_attr_for_testing()
    except:
        return
    
    if attrs is None:
        return

    # set the attributes
    for attr_name in attrs.keys():
        attr = attrs[attr_name]
        n_el = len(attr)
        n_envs = venv.num_envs
        for i in range(n_envs):
            _i = i % n_el 
            # repeat the attributes if the number of attributes is 
            # less than the number of envs
            venv.set_attr(attr_name, attr[_i], indices=i)
    
    print("Attributes are loaded for testing.")
    return

if __name__ == "__main__":
    a = np.random.randint(0, 4, size=(2,10))
    print(f"Original: \n{a}")
    print(f"Converted: \n{assginment_rsc_to_stat_2d(a, n_stations = 8)}")
