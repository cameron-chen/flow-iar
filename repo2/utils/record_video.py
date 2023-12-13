import argparse
import os
import sys

import pygame
from pyvirtualdisplay import Display
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecEnv, VecVideoRecorder

sys.path.insert(0, '/home/changyu/experiment/oregon/constrained_rl/')
from envs import gym_seqssg, pistonball, pursuit
from envs.util import create_test_env
from policies.policy import get_policy_id
from utils.exp_manager import ExperimentManager
from utils.util import (ALGOS, StoreDict, args_parser, dict2obj,
                        get_saved_hyperparams)


def set_up_algo(args, rl_algo: BaseAlgorithm, env: VecEnv):
    """
    Set up the algorithm and the corresponding policy.
    """
    if args.policy == "mask":
        rl_algo.policy.upd_mask_fn(env.envs[0].gen_mask_from_obs)

if __name__ == "__main__":  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="log")
    parser.add_argument("-o", "--output-folder", help="Output folder", type=str)
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('--policy', type=str, default='MlpPolicy')
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--exp-id", help="Experiment ID (folder name of the experiment)", required=True, type=str)
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    args = parser.parse_args()

    env_id = args.env
    algo = args.algo
    folder = args.folder
    policy_id = get_policy_id(args)
    video_folder = args.output_folder
    seed = args.seed
    video_length = args.n_timesteps
    n_envs = args.n_envs
    load_best = args.load_best
    load_checkpoint = args.load_checkpoint

    # Sanity checks
    log_path = os.path.join(folder, env_id.lower(), algo.lower(), policy_id.lower(), args.exp_id)
    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    found = False
    for ext in ["zip"]:
        model_path = os.path.join(log_path, "policies", f"{env_id.lower()}.{ext}")
        found = os.path.isfile(model_path)
        name_prefix = f"final-model-{algo}-{env_id}"
        if found:
            break

    if args.load_best:
        model_path = os.path.join(log_path, "policies", "best_model.zip")
        found = os.path.isfile(model_path)
        name_prefix = f"best-model-{algo}-{env_id}"

    if args.load_checkpoint is not None:
        model_path = os.path.join(log_path, "policies", f"rl_model_{args.load_checkpoint}_steps.zip")
        found = os.path.isfile(model_path)
        name_prefix = f"checkpoint-{args.load_checkpoint}-{algo}-{env_id}"

    if not found:
        raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")

    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    set_random_seed(args.seed)

    is_atari = ExperimentManager.is_atari(env_id)
    is_pygame = "Pursuit" in args.env

    stats_path = os.path.join(log_path, "config")
    hyperparams, stats_path = get_saved_hyperparams(stats_path)

    hyperparams["log_wandb"] = False
    hyperparams["device"] = 'auto'

    # load env_kwargs if existing
    _, __, env_kwargs, ___, ____ = args_parser(dict2obj(hyperparams))
    
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    env = create_test_env(
        env_id,
        n_envs=n_envs,
        stats_path=os.path.join(log_path, "policies"),
        seed=seed,
        log_dir=None,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
        vec_normalize_final= (not args.load_best),
    )

    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    print(f"Loading {model_path}")

    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)
    set_up_algo(args, model, env)

    obs = env.reset()

    # Deterministic by default except for atari games
    stochastic = args.stochastic or is_atari and not args.deterministic
    deterministic = not stochastic

    # Pygame init
    if is_pygame: 
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        screen = pygame.display.set_mode((1400, 900))

    if video_folder is None:
        video_folder = os.path.join(log_path, "videos")

    # Note: apparently it renders by default
    env = VecVideoRecorder(
        env,
        video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix=name_prefix,
    )

    env.reset()
    display = Display(visible=0, size=(1400, 900))
    display.start()
    try:
        for _ in range(video_length + 1):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, _, _ = env.step(action)
            if not args.no_render:
                env.render()
        env.close()
    except KeyboardInterrupt:
        pass
    finally:
        # Release everything if job is finished
        display.stop()
        if is_pygame:
            pygame.quit()