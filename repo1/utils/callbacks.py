import itertools
import os
from typing import Iterable, Optional, Union

import gym
import numpy as np
from stable_baselines3.common.callbacks import (BaseCallback,
                                                CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization

# from envs.gym_seqssg.base import SeqSSG           #no-numba
from envs.gym_seqssg.game_env_base import SeqSSG  # numba-supported


class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize wrapper every ``save_freq`` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: Optional[str] = None, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            else:
                path = os.path.join(self.save_path, "vecnormalize.pkl")
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print(f"Saving VecNormalize to {path}")
        return True

class CheckpointCallback_V2(CheckpointCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    
    Add capability of excluding parameters to save.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``
      
    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param verbose:
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", 
                 exclude: Optional[Iterable[str]] = None, verbose: int = 0):
        super().__init__(save_freq, save_path, name_prefix, verbose)
        self.exclude = exclude
        
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path, exclude=self.exclude)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
        return True

class EvalCallback_V2(EvalCallback):
    """
    Callback for evaluating an agent.

    Add capability of excluding parameters to save.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        exclude: Optional[Iterable[str]] = None,
    ):
        super().__init__(eval_env, callback_on_new_best, callback_after_eval, 
                        n_eval_episodes, eval_freq, log_path, best_model_save_path, 
                        deterministic, render, verbose, warn)
        self.exclude = exclude

    def _on_step(self) -> bool:

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    )

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"),
                                    exclude=self.exclude)
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

class UpdMaskFnCallback(BaseCallback):
    """Callback for updating the mask_fn of the policy at the beginning of training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_training_start(self) -> None:
        self.model.policy.upd_mask_fn(self.training_env.envs[0].gen_mask_from_obs)

    def _on_step(self) -> bool:
        return True

class CstrEnvCallback(BaseCallback):
    """
    Callback for SeqSSG env to retrieve game status, such as `valid`.
    """
    def __init__(
        self, 
        verbose: int = 0, 
        eval_freq: int = 2000, 
        buf_size: int = 300,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.buf_size = buf_size 
        self.val_rate = []
        self.eval_timesteps = 0

    def _init_callback(self) -> None:
        n_envs = self.model.env.num_envs
        self.valids = -np.ones((n_envs, self.buf_size), dtype=np.float32)
        self.idx_pos = np.zeros(n_envs, dtype=np.int32)

    def _on_step(self) -> bool:
        # Collect game status
        infos = self.training_env.buf_infos
        dones = self.training_env.buf_dones
        for i, info in enumerate(infos):
            i_pos = self.idx_pos[i]
            self.valids[i, i_pos] = info['valid']
            self.idx_pos[i] += 1
            if dones[i]:
                val = self.valids[i, :i_pos+1]
                self.val_rate.append(val.sum()/val.shape[0])
                self.valids[i] = -np.ones_like(self.valids[i], dtype=np.float32)
                self.idx_pos[i] = 0

        # Evaluate and log
        if self.num_timesteps-self.eval_timesteps >= self.eval_freq:
            val_rate = safe_mean(self.val_rate)
            self.logger.record("rollout/val_rate_mean", val_rate)
            if self.verbose > 1:
                print(f"val_rate: {val_rate:.4f}")
            self.val_rate = []
            self.eval_timesteps = self.num_timesteps
        
        return True

class SeqSSGCallback(BaseCallback):
    """
    Callback for SeqSSG env to retrieve game status, such as
        `protected`, `valid`, and `def_util_ori`.

    Another function is to check if the environment is stuck 
        (has no feasible actions and stuck at certain state).
    """
    def __init__(
        self, 
        verbose: int = 0, 
        eval_freq: int = 2000, 
        check_freq: int = int(4e5),
        buf_size: int = 300,
        max_n_cand_act: int = int(5e4),
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.check_freq = check_freq
        self.buf_size = buf_size
        self.max_n_cand_act = max_n_cand_act
        self.prot_rate = []
        self.val_rate = []
        self.rets = []
        self.eval_timesteps = 0
        self.check_timesteps = 0
        self.mask_fn = None
        self.cand_act = None

    def _init_callback(self) -> None:
        assert isinstance(self.training_env.envs[0].unwrapped, SeqSSG)
        n_envs = self.model.env.num_envs
        self.protecteds = -np.ones((n_envs, self.buf_size), dtype=np.float32)
        self.valids = -np.ones((n_envs, self.buf_size), dtype=np.float32)
        self.rewards = -np.ones((n_envs, self.buf_size), dtype=np.float32)*np.inf
        self.idx_pos = np.zeros(n_envs, dtype=np.int32)

        # Get mask_fn and cand_act
        raw_env = self.training_env.envs[0].unwrapped
        self.mask_fn = raw_env.gen_mask_from_obs
        if np.prod(raw_env.action_space.nvec) > self.max_n_cand_act:
            self.cand_act = np.array([raw_env.action_space.sample() 
                for _ in range(self.max_n_cand_act)])
        else:
            self.cand_act = np.array(list(itertools.product(*[
                range(dim) for dim in raw_env.action_space.nvec])))

    def _on_step(self) -> bool:
        # Collect game status
        infos = self.training_env.buf_infos
        dones = self.training_env.buf_dones
        for i, info in enumerate(infos):
            i_pos = self.idx_pos[i]
            self.protecteds[i, i_pos] = info['protected']
            self.valids[i, i_pos] = info['valid']
            self.rewards[i, i_pos] = info['rew_ori']
            self.idx_pos[i] += 1
            if dones[i]:
                prot = self.protecteds[i, :i_pos+1]
                val = self.valids[i, :i_pos+1]
                rew = self.rewards[i, :i_pos+1]
                self.prot_rate.append(prot.sum()/prot.shape[0])
                self.val_rate.append(val.sum()/val.shape[0])
                self.rets.append(rew.sum())
                self.protecteds[i] = -np.ones_like(self.protecteds[i], dtype=np.float32)
                self.valids[i] = -np.ones_like(self.valids[i], dtype=np.float32)
                self.rewards[i] = -np.ones_like(self.rewards[i], dtype=np.float32)*np.inf
                self.idx_pos[i] = 0

        # Evaluate and log
        if self.num_timesteps-self.eval_timesteps >= self.eval_freq:
            prot_rate = safe_mean(self.prot_rate)
            val_rate = safe_mean(self.val_rate)
            ret = safe_mean(self.rets)
            self.logger.record("rollout/prot_rate_mean", prot_rate)
            self.logger.record("rollout/val_rate_mean", val_rate)
            self.logger.record("rollout/ep_ori_rew_mean", ret)
            if self.verbose > 1:
                print(f"prot_rate: {prot_rate:.4f}, val_rate: {val_rate:.4f}, ret: {ret:.4f}")
            self.prot_rate = []
            self.val_rate = []
            self.rets = []
            self.eval_timesteps = self.num_timesteps
        
        return True

    def _on_rollout_end(self) -> None:
        if self.num_timesteps-self.check_timesteps>=self.check_freq:
            # Obtain current observation
            obs = self.model._last_obs

            # Check if valid actions exist 
            masks = np.array([self.mask_fn(o, self.cand_act) for o in obs])
            if np.any(masks.sum(axis=1)==0):
                raise RuntimeError("No valid actions exist, the environment is stuck.")

            self.check_timesteps = self.num_timesteps
