import matplotlib.pyplot as plt
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure, TensorBoardOutputFormat
from stable_baselines3.common.utils import obs_as_tensor
from torchinfo import summary

from utils.util import viz_weight_grad_norm, viz_weight_norm

from .util import stat_flow_act_and_log_prob


class UpdateFlowNetCallback(BaseCallback):
    """Update the flow net in the `FlowPolicy` before collecting rollouts.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
    
    def _init_callback(self) -> None:
        # Create dummy callback
        dummy_callback = DummyCallback()
        dummy_callback.init_callback(self.model)
        
        # Collect rollouts
        print('Collecting rollouts in obs_buffer for flow net updating ...')
        n_steps = self.model.n_steps
        buffer_size = self.model.policy.obs_buffer.buffer_size
        while not self.model.policy.obs_buffer.full:
            self.model.collect_rollouts(
                self.model.env, dummy_callback, self.model.rollout_buffer, 
                n_rollout_steps=n_steps)
            self.model.policy.update_obs_buffer(self.model.rollout_buffer.observations)
        
        print(f"Obs_buffer is full. Size: {buffer_size}.")

        # Reset num_timesteps,
        #   maybe reset env (self._last_obs, self._last_episode_starts, 
        #   self._last_original_obs), policy buffer
        self.model.num_timesteps = 0

        # Pretrain flow net
        print('Pretraining flow net ...')
        ## Setup
        self.model.policy.set_training_mode(True)
        num_envs = self.model.env.num_envs
        batch_size = self.model.policy.batch_size_flow_updating
        n_iters = self.model.policy.n_iters_flow_pretraining or max((5 * buffer_size // (num_envs * batch_size)), 10)
        ## Updating
        for i in range(n_iters):
            obs = self.model.policy.obs_buffer.sample(batch_size)
            obs_tensor = obs_as_tensor(obs, self.model.device)
            self.model.policy.flow_net_updating(obs_tensor)
        self.model.policy.set_training_mode(False)

        print('Done pretraining flow net.')

    def _on_rollout_start(self) -> None:
        # self.model: RL algo
        if not self.model.policy.obs_buffer.full: return

        # Switch to train mode (this affects batch norm / dropout)
        self.model.policy.set_training_mode(True)

        # Retrieve states
        batch_size = self.model.policy.batch_size_flow_updating
        obs = self.model.policy.obs_buffer.sample(batch_size)
        obs_tensor = obs_as_tensor(obs, self.model.device)

        # Flow net updating
        loss = self.model.policy.flow_net_updating(obs_tensor)

        # Switch to eval mode (this affects batch norm / dropout)
        self.model.policy.set_training_mode(False)

        # Log
        self.logger.record('train/flow_net_elbo', loss.item())

    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        # self.model.rollout_buffer: rollout buffer
        self.model.policy.update_obs_buffer(self.model.rollout_buffer.observations)

class ActCorrCallback(BaseCallback):
    """
    Action correction callback. Update the flow net, in the `FlowPolicy`, 
        to generate valid actions before call `UpdateFlowNetCallback`.
    """
    def __init__(self, verbose: int = 0, corr_prot: str = "flow"):
        super().__init__(verbose)
        self.corr_prot = corr_prot

    def _on_rollout_start(self) -> None:
        if self.corr_prot == "flow_joint": # Do nothing if `flow_joint` mode
            return 
        
        if self.corr_prot == "val_joint": 
            self._update_validator()
        elif self.corr_prot == "val":
            self._corr_by_val()
        elif self.corr_prot == "flow":
            self._corr_by_flow()
        else:
            raise ValueError(f"Unknown correction protocol: {self.corr_prot}, expected 'val_joint', 'val' or 'flow'.")

    def _on_step(self) -> bool:
        return True
    
    def _update_validator(self): 
        """Update the validator."""
        raise NotImplementedError("Not implemented yet.")

    def _corr_by_val(self):
        """Correct actions by validator.
        """
        # self.model: RL algo
        # self.training_env: training env
        if not self.model.policy.obs_buffer.full: return

        # Switch to train mode (this affects batch norm / dropout)
        self.model.policy.set_training_mode(True)

        # ========= Start to Edit =========
        # Retrieve states 
        batch_size = self.model.policy.batch_size_flow_updating
        obs = self.model.policy.obs_buffer.sample(batch_size)
        obs_tensor = obs_as_tensor(obs, self.model.device)

        # Flow net updating
        val_loss, flow_loss = self.model.policy.act_corr_val(obs_tensor,
            self.training_env.envs[0].val_inv_act_gen, 
            self.training_env.envs[0].act_check)

        # ======== End of Editing ========
        # Switch to eval mode (this affects batch norm / dropout)
        self.model.policy.set_training_mode(False)

        # Log
        self.logger.record('train/val_loss', val_loss)
        self.logger.record('train/val_flow_loss', flow_loss)

    def _corr_by_flow(self):
        """Correct actions by flow net.
        """
        # self.model: RL algo
        # self.training_env: training env
        if not self.model.policy.obs_buffer.full: return

        # Switch to train mode (this affects batch norm / dropout)
        self.model.policy.set_training_mode(True)

        # ========= Start to Edit =========
        # Retrieve states 
        batch_size = self.model.policy.batch_size_flow_updating
        obs = self.model.policy.obs_buffer.sample(batch_size)
        obs_tensor = obs_as_tensor(obs, self.model.device)

        # Flow net updating
        self.model.policy.act_corr_flow(obs_tensor,
            self.training_env.envs[0].act_check)

        # ======== End of Editing ========
        # Switch to eval mode (this affects batch norm / dropout)
        self.model.policy.set_training_mode(False)

class LogFlowNetDistCallback(BaseCallback):
    """Log the behavior of flow net distribution.

    Verifies:
        EMD: EMD between the empirical distribution and the distribution 
            parameterized by flow network (flow distribution) over timesteps.
        Errorbar: Plot the empirical distribiton and the flow distribution by error bar plog.
    """
    def __init__(
        self, 
        verbose: int = 0, 
        batch_size:int=1024,
        n_steps_eval: int = 10000,
    ):
        super().__init__(verbose)
        self.batch_size = batch_size
        self.n_steps_eval = n_steps_eval
        self.n_timesteps_eval = 0
        self.counter_eval = 1
        self.fig = None

    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        if self.num_timesteps-self.n_timesteps_eval > self.n_steps_eval:
            # Obtain observation: 1 data point and extend to batch_size (keep obs the same)
            obs_single = self.model.rollout_buffer.observations[0,0].copy().reshape(1,-1)
            obs = np.repeat(obs_single, self.batch_size, axis=0)

            # Sample actions
            with th.no_grad():
                obs_tensor = obs_as_tensor(obs, self.model.device)
                acts, _, log_prob = self.model.policy(obs_tensor)
            
            # Process actions: stats
            acts = acts.cpu().numpy()
            assert len(acts.shape) < 2, f"Only support 1-D actions. The shape of acts is {acts.shape}."
            log_prob = log_prob.cpu().numpy()
            stat, emd = stat_flow_act_and_log_prob(acts, log_prob)
            
            # Log
            self.logger.record("eval/emd_act_dist", emd)
            if self.counter_eval % 10 == 0:
                # Plot
                fig = plt.figure(figsize=(8,6))
                fig.add_subplot().errorbar(stat[:,0], stat[:,1], stat[:,2], fmt='ok', label='Flow dist')
                plt.scatter(stat[:,0], stat[:,3], marker='_', color='r', s=400, label='Emp dist')
                plt.style.use('seaborn-deep')
                plt.legend()

                self.logger.record("eval/act_dist_errorbar_plot", Figure(fig, close=True), 
                                   exclude=("stdout", "log", "json", "csv"))
            self.counter_eval += 1
            self.n_timesteps_eval = self.num_timesteps

class TrackModelGradCallback(BaseCallback):
    """Track the weight and gradient of the model.
    """
    def __init__(
        self, 
        verbose: int = 0, 
        n_steps_track:int=100000,
        pairs4replace=[],
    ):
        super().__init__(verbose)
        self.n_steps_track = n_steps_track
        self.n_timesteps_track = 0
        self.pairs4replace = pairs4replace

    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_start(self) -> None:
        plt.close('all')

    def _on_rollout_end(self) -> None:
        if self.num_timesteps-self.n_timesteps_track > self.n_steps_track:
            components_to_track = [['flow_net','posterior'], ['mlp_extractor']]
            for comp_str in components_to_track:
                self.logger.record(
                    f"eval/{'_'.join(comp_str)}/weight_norm", 
                    Figure(viz_weight_norm(
                        self.model.policy, lay_names=comp_str,
                        pairs4replace=self.pairs4replace), close=True),
                    exclude=("stdout", "log", "json", "csv"))
                self.logger.record(
                    f"eval/{'_'.join(comp_str)}/weight_grad_norm",
                    Figure(viz_weight_grad_norm(
                        self.model.policy, lay_names=comp_str,
                        pairs4replace=self.pairs4replace), close=True),
                    exclude=("stdout", "log", "json", "csv"))
            self.n_timesteps_track = self.num_timesteps

class LogModelStructureCallback(BaseCallback):

    def _on_training_start(self):
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))
        
        # log the model structure
        ## Setup
        batch_size = self.model.policy.batch_size_flow_updating
        obs = self.model.policy.obs_buffer.sample(batch_size)
        obs_tensor = obs_as_tensor(obs, self.model.device)
        ## log the structure of the policy
        self.tb_formatter.writer.add_graph(self.model.policy, obs_tensor)
        self.tb_formatter.writer.flush()
        print("Logging the model structure...done")

    def _on_step(self) -> bool:
        return True

class ModelSummaryCallback(BaseCallback):
    """Save the model in onnx format.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        return True

    def _on_training_start(self) -> None:
        # Setup
        batch_size = self.model.policy.batch_size_flow_updating
        obs = self.model.policy.obs_buffer.sample(batch_size)
        obs_shape = obs.shape
        # Print
        summary(self.model.policy, obs_shape)

class DummyCallback(BaseCallback):
    """Dummy callback.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True
