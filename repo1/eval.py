"""Script from rl-baselines3-zoo."""

import argparse
import glob
import importlib
import itertools
import os
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from einops import repeat
from gym import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecEnv

from envs import gym_seqssg, pistonball, pursuit
from envs.util import create_test_env
from policies.policy import get_policy_id
from utils.exp_manager import ExperimentManager
from utils.util import (ALGOS, StoreDict, args_parser, dict2obj,
                        discrete_var_cluster, discrete_var_hist,
                        get_saved_hyperparams, op_on_obs, set_seeds)


def main():  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="log")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('--policy', type=str, default='MlpPolicy')
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (folder name of the experiment)", required=True, type=str)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
        "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument(
        "--load-last-checkpoint",
        action="store_true",
        default=False,
        help="Load last checkpoint instead of last model if available",
    )
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import (e.g. gym_minigrid)",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument("--viz-action-dist", type=eval, default=False)
    parser.add_argument("--load_attr-for-testing", action="store_true", default=False, 
                        help="Load attr for testing. E.g. fix the initial state of the environment.")
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    folder = args.folder
    policy_id = get_policy_id(args)

    # Sanity checks
    log_path = os.path.join(folder, env_id.lower(), algo.lower(), policy_id.lower(), args.exp_id)
    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    found = False
    for ext in ["zip"]:
        model_path = os.path.join(log_path, "policies", f"{env_id.lower()}.{ext}")
        found = os.path.isfile(model_path)
        if found:
            break

    if args.load_best:
        model_path = os.path.join(log_path, "policies", "best_model.zip")
        found = os.path.isfile(model_path)

    if args.load_checkpoint is not None:
        model_path = os.path.join(log_path, "policies", f"rl_model_{args.load_checkpoint}_steps.zip")
        found = os.path.isfile(model_path)

    if args.load_last_checkpoint:
        checkpoints = glob.glob(os.path.join(log_path, "policies", "rl_model_*_steps.zip"))
        if len(checkpoints) == 0:
            raise ValueError(f"No checkpoint found for {algo} on {env_id}, path: {log_path}")

        def step_count(checkpoint_path: str) -> int:
            # path follow the pattern "rl_model_*_steps.zip", we count from the back to ignore any other _ in the path
            return int(checkpoint_path.split("_")[-2])

        checkpoints = sorted(checkpoints, key=step_count)
        model_path = checkpoints[-1]
        found = True

    if not found:
        raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")

    print(f"Loading {model_path}")

    # Off-policy algorithm only support one env for now
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    is_atari = ExperimentManager.is_atari(env_id)

    stats_path = os.path.join(log_path, "config")
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    hyperparams["log_wandb"] = False
    hyperparams['device'] = args.device

    # load env_kwargs if existing
    _, __, env_kwargs, ___, ____ = args_parser(dict2obj(hyperparams))

    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    env = create_test_env(
        env_id,
        n_envs=args.n_envs,
        stats_path=os.path.join(log_path, "policies"),
        seed=args.seed,
        log_dir=log_dir,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
        vec_normalize_final= (not args.load_best),
        load_attr=args.load_attr_for_testing,
    )
    # HACK: to avoid the space check error when load the model
    if 'wolp' in algo:
        # FIXME: still cannot evaluate wolp series model
        _max_action = env.action_space.nvec[0].item()-1
        _shape = env.action_space.shape
        env.action_space = spaces.Box(low=0.0, high=_max_action, shape=_shape, dtype=np.float32)

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

    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, device=args.device, **kwargs)
    set_up_algo(args, model, env)

    obs = env.reset()

    # Deterministic by default except for atari games
    stochastic = args.stochastic or is_atari and not args.deterministic
    deterministic = not stochastic

    state = None
    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    if args.viz_action_dist: act_ref = np.array(list(itertools.product(*[
        range(dim) for dim in env.action_space.nvec])))
    # For HER, monitor success rate
    successes = []
    try:
        for i in range(args.n_timesteps):
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, infos = env.step(action)
            if not args.no_render:
                env.render("human")

            episode_reward += reward[0]
            ep_len += 1

            if i in [10,11,12] and args.viz_action_dist:
                setup_and_plot(log_path, obs, model, act_ref, 
                               f'act_dist_{i:02d}', deterministic=deterministic)

            if args.n_envs == 1:
                # For atari the return reward is not the atari score
                # so we have to get it from the infos dict
                if is_atari and infos is not None and args.verbose >= 1:
                    episode_infos = infos[0].get("episode")
                    if episode_infos is not None:
                        print(f"Atari Episode Score: {episode_infos['r']:.2f}")
                        print("Atari Episode Length", episode_infos["l"])

                if done and not is_atari and args.verbose > 0:
                    # NOTE: for env using VecNormalize, the mean reward
                    # is a normalized reward when `--norm_reward` flag is passed
                    print(f"Episode Reward: {episode_reward:.2f}")
                    print("Episode Length", ep_len)
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(ep_len)
                    episode_reward = 0.0
                    ep_len = 0
                    state = None

                # Reset also when the goal is achieved when using HER
                if done and infos[0].get("is_success") is not None:
                    if args.verbose > 1:
                        print("Success?", infos[0].get("is_success", False))

                    if infos[0].get("is_success") is not None:
                        successes.append(infos[0].get("is_success", False))
                        episode_reward, ep_len = 0.0, 0

    except Exception as e:
        print(e)
        traceback.print_exc()
    except KeyboardInterrupt:
        pass

    if args.verbose > 0 and len(successes) > 0:
        print(f"Success rate: {100 * np.mean(successes):.2f}%")

    if args.verbose > 0 and len(episode_rewards) > 0:
        print(f"{len(episode_rewards)} Episodes")
        print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

    if args.verbose > 0 and len(episode_lengths) > 0:
        print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

    env.close()

def set_up_algo(args, rl_algo: BaseAlgorithm, env: VecEnv):
    """
    Set up the algorithm and the corresponding policy.
    """
    if args.policy == "mask":
        rl_algo.policy.upd_mask_fn(env.envs[0].gen_mask_from_obs)

def plot_action_dist(
    path: str,
    obs: np.ndarray, 
    model: BaseAlgorithm, 
    act_ref: np.ndarray,
    n_samples: int = 200,
    deterministic: bool = False,
):
    """
    Plot the action distribution of the policy.
    """
    # Extend the observation
    obs_ext = op_on_obs(repeat, obs, pattern="b ... -> (b r) ...", r=n_samples)
    # obs_ext = repeat(obs, "b ... -> (b r) ...", r=n_samples)
    
    # Sample actions
    actions, _ = model.predict(obs_ext, deterministic=deterministic)

    # Plot the action distribution
    fig_dist, var_stat = discrete_var_hist(actions, act_ref)
    fig_dist_cluster = discrete_var_cluster(actions)

    # Save
    fig_dist.savefig(os.path.join(path, "action_dist.png"))
    fig_dist_cluster.savefig(os.path.join(path, "action_dist_cluster.png"))
    save_var_stat(os.path.join(path, "action_dist.txt"), var_stat, act_ref)

    # Close
    plt.close(fig_dist)
    plt.close(fig_dist_cluster)

def save_var_stat(path: str, var_stat: np.ndarray, act_ref: np.ndarray):
    """
    Save the variable statistics.
    """
    sorted_idx = np.argsort(-var_stat)

    with open(path, "w") as f:
        f.write("Index\tAction\tNumber")
        # for i, (act, num) in enumerate(zip(act_ref, var_stat)):
        for i in sorted_idx:
            f.write(f"\n{i:06d}\t{act_ref[i]}\t{var_stat[i]}")
    
def setup_and_plot(
    root: str,
    obs: np.ndarray, 
    model: BaseAlgorithm, 
    act_ref: np.ndarray,
    folder_name: str = None,
    n_samples: int = 200,
    deterministic: bool = False,
):
    assert obs.shape[0] == 1, "Only support single observation"
    assert os.path.exists(root), "Root directory does not exist"

    root = os.path.join(root, 'viz')
    if folder_name is not None:
        path = os.path.join(root, folder_name)
    else:
        path = root
    os.makedirs(path, exist_ok=True)
    
    plot_action_dist(path, obs, model, act_ref, n_samples, deterministic)

if __name__ == "__main__":
    main()
