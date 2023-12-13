"""Uses RL to train a policy from scratch, saving rollouts and policy."""

import argparse

import numpy as np
import psutil
import torch as th

from envs.env import add_env_args, get_env_id
from policies.policy import add_policy_args, get_policy, get_policy_id
from utils.exp_manager import ExperimentManager
from utils.rl_algo.util import add_rl_args, get_rl_algo, get_rl_id
from utils.util import args_parser, set_seeds

if __name__ =='__main__':
    ###########
    ## Setup ##
    ###########
    parser = argparse.ArgumentParser()
    add_env_args(parser)
    add_policy_args(parser)
    add_rl_args(parser)
    args = parser.parse_args()
    if args.seed < 0:
        # Seed but with a random one
        args.seed = np.random.randint(2**32 - 1, dtype="int64").item()
    set_seeds(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)
    
    if args.cpu_set is not None:
        psutil.Process().cpu_affinity(args.cpu_set)
        print(f"Setting process affinity to {args.cpu_set}")
    
    print("=" * 10, args.env_id, "=" * 10)
    print(f"Seed: {args.seed}")

    #########################
    ## Specify environment ##
    #########################

    env_id = get_env_id(args)

    ####################
    ## Specify policy ##
    ####################

    policy_cls = get_policy(args)
    policy_id = get_policy_id(args)

    #######################
    ## Specify algorithm ##
    #######################

    rl_cls = get_rl_algo(args)
    rl_id = get_rl_id(args)

    ##############
    ## Training ##
    ##############
    assert 'Pistonball' in env_id, "Note: this script is specificly for PistonballCstr env."

    rl_kwargs, policy_kwargs, env_kwargs, normalize_kwargs, wandb_kwargs = args_parser(args)
    exp_manager = ExperimentManager(
        args=args,
        total_timesteps=args.total_timesteps,
        rl_cls=rl_cls,
        rl_kwargs=rl_kwargs,
        policy_cls=policy_cls,
        policy_kwargs=policy_kwargs,
        n_envs=args.n_envs,
        env_kwargs=env_kwargs,
        normalize=args.normalize,
        normalize_kwargs=normalize_kwargs,
        policy_save_interval=args.policy_save_interval,
        policy_eval_interval=args.policy_eval_interval,
        n_eval_envs=args.n_eval_envs,
        n_eval_episodes=args.n_eval_episodes,
        seed=args.seed,
        env_id=env_id, 
        rl_id=rl_id, 
        policy_id=policy_id,
        wandb_kwargs=wandb_kwargs,
        log_interval=args.log_interval,
        max_episode_steps=args.max_episode_steps,
        vec_env_type=args.vec_env_type,
        convert_act=args.convert_act if "ERSEnv" in args.env_id else False,
        format_str=args.format_str,
        log_flow_dist=args.log_flow_dist if args.policy=='flow' else False,
        log_weight_grad=args.log_weight_grad,
        log_model_structure=args.log_model_structure,
        model_summary=args.model_summary,
        has_act_corr=args.has_act_corr if args.policy=='flow' else False,
        act_corr_prot=args.act_corr_prot if args.policy=='flow' else None,
        verbose=args.verbose
    )
    exp_manager.learn()
    exp_manager.save_trained_model()
