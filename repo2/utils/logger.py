"""Logging for quantitative metrics and free-form text."""

import os
import time
from typing import Any, Mapping, Optional, Sequence, Tuple

import wandb
from stable_baselines3.common.logger import Logger, configure

from .util import clean_dict, get_exp_idx

no_log_keys = [ 'log_wandb', 'policy_save_interval',
                'policy_eval_interval', 'device', 
                'project', 'run_notes']

def make_log_dir(
    env_id: str,
    rl_id: str,
    policy_id: str,
    name: str = None,
    log_base: str = './log',
) -> str:
    """Make log_dir and create log folder.

    Args:
        log_dir (optional): The directory to log to.

    Return:
        log_dir
    """
    if not name:
        _log_parent_path = os.path.join(log_base, env_id, rl_id, policy_id)
        exp_idx = get_exp_idx(_log_parent_path) 
        name = "{:03d}_{}".format(exp_idx,time.strftime("%Y-%m-%d_%H-%M-%S"))
    log_dir = os.path.join(log_base, env_id, rl_id, policy_id, name)

    os.makedirs(log_dir, exist_ok=True)

    return log_dir

def setup_logging(
    env_id: str,
    rl_id: str,
    policy_id: str,
    format_strs: Optional[Sequence[str]] = None,
    args = None,
    log_wandb: bool = False,
    project: str = None,
    run_notes: str = None,
) -> Tuple[Logger, str]:
    """Build the Stable Baselines logger.

    Args:
        env_id: environment id.
        rl_id: RL algorithm id.
        policy_id: Policy module id.
        format_strs: The types of formats to log to.

    Returns:
        The configured logger and `log_dir`.
    """
    log_dir = make_log_dir(env_id, rl_id, policy_id)
    
    if log_wandb:
        assert project is not None, "Wandb project name must be provided."
        assert args is not None, "Wandb config must be provided."
        args_dict = clean_dict(vars(args), keys=no_log_keys)
        wandb.tensorboard.patch(root_logdir=os.path.join(log_dir, 'log'))
        wandb.init(config=args_dict, project=project, id=f'{os.path.basename(log_dir)}', 
                       dir=log_dir, notes=run_notes)

    # wandb.init should be called before configure, otherwise it will not 
    #   track tensorboard.
    logger = configure(
        folder=os.path.join(log_dir, 'log'), 
        format_strings=format_strs
    )

    return logger, log_dir

