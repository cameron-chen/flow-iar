import stable_baselines3

from .a2c import A2C_Corr, A2C_Flow, A2C_Rej
from .wol_ddpg import Wol_DDPG, Wol_DDPG_Rej


def add_rl_args(parser):
    # RL algo params
    ## Shared
    parser.add_argument('--rl_algo', type=str, default='A2C')
    parser.add_argument('--n_steps', type=int, default=-1,
        help='Number of transitions to collect from one env (-1 to use default).'
    )
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=str, default="3e-4")
    parser.add_argument('--gamma', type=float, default=0.99)
    ## On-policy
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--ent_coef', type=float, default=0.0)
    parser.add_argument('--min_reg_coef', type=float, default=0.0)
    parser.add_argument('--max_reg_coef', type=float, default=0.0)
    parser.add_argument('--gae_lambda', type=float, default=1.0)
    parser.add_argument('--noise_eps', type=float, default=-1.0)
    ## Off-policy
    parser.add_argument('--buffer_size', type=int, default=int(1e6))
    parser.add_argument('--learning_starts', type=int, default=100)
    parser.add_argument('--df_max_actions', default=200000, type=int, help='max actions')
    parser.add_argument('--k_ratio', default=0.1, type=float, help='')
    ### Wol-DDPG
    parser.add_argument('--flann_mode', type=str, default='medium', choices=['medium', 'slow'])

    # Training params
    parser.add_argument('--total_timesteps', type=float, default=1e6)
    parser.add_argument('--normalize', type=eval, default=False)
    parser.add_argument('--normalize_reward', type=eval, default=True)
    parser.add_argument('--normalize_obs', type=eval, default=True)
    parser.add_argument('--gamma_norm_env', type=float, default=0.99)
    parser.add_argument('--policy_save_interval', type=int, default=0)
    parser.add_argument('--policy_eval_interval', type=int, default=0)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--format_str', type=str, default='tensorboard_stdout')
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--cpu-set", nargs='*', default=None, type=int, help="List[int] a list of CPUs to use")
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument("--min_n_samples_inv_rej", default=-1, type=int,
        help="Minimal number of samples for invalid action rejection A2C")
    parser.add_argument("--n_jobs", default=-1, type=int, 
        help="Number of jobs for parallelization")

    # Evaluation params
    parser.add_argument('--n_eval_episodes', type=int, default=10)

    # Log params
    parser.add_argument('--log_wandb', type=eval, default=False)
    parser.add_argument('--log_interval', type=int, default=-1)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument('--project', type=str, default='constrained_rl')
    parser.add_argument('--run_notes', type=str, default='')

def get_rl_id(args):
    return args.rl_algo

def get_rl_algo(args):
    if args.rl_algo == "A2C":
        if getattr(args, 'has_act_corr', False) and 'joint' in getattr(args, 'act_corr_prot', ''):
            return A2C_Corr
        return stable_baselines3.A2C
    elif args.rl_algo == "A2C_Rej":
        return A2C_Rej
    elif args.rl_algo == "A2C_Flow":
        return A2C_Flow
    elif args.rl_algo == "PPO":
        return stable_baselines3.PPO
    elif "Wolp" in args.rl_algo:
        if args.rl_algo == "Wolp-Rej":
            return Wol_DDPG_Rej
        elif args.rl_algo == "Wolp":
            return Wol_DDPG
        else:
            raise ValueError("Unknown Wolpertinger algorithm: {}".format(args.rl_algo))
    elif args.rl_algo == "DDPG":
        return stable_baselines3.DDPG
    else:
        raise NotImplementedError()
