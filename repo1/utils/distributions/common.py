from typing import List, Union

import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.distributions import CategoricalDistribution
from torch.distributions import Categorical


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            device = self.masks.device
            self.masks = masks.type(th.BoolTensor).to(device)
            logits = th.where(self.masks, logits, th.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        device = self.masks.device
        p_log_p = self.logits * self.probs
        p_log_p = th.where(self.masks, p_log_p, th.tensor(0.).to(device))
        return -p_log_p.sum(-1)

class CategoricalMaskDistribution(CategoricalDistribution):
    """ 
    Categorical distribution for discrete actions with action mask support.

    Args:
        action_dim: Number of discrete actions

    Note:
        This distribution is used for discrete actions or multidiscrete actions.
            For the latter, the distribution models the joint distribution by mapping
            an action to an index of action combinations. E.g. action_dim = [3, 3, 2],
            there are 18 possible actions. 
    """

    def __init__(self, action_dim: Union[int, List[int]]):
        super().__init__(1) # Hack: use 1 as placeholder
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
            it will be the logits of the Categorical distribution.
            You can then get probabilities using a softmax.

        Args: 
            latent_dim: Dimension of the last layer
                of the policy network (before the action layer)
        """
        action_logits = nn.Linear(latent_dim, np.prod(self.action_dim).item())
        return action_logits

    def proba_distribution(
        self, 
        action_logits: th.Tensor,
        masks: th.Tensor
    ) -> "CategoricalMaskDistribution":
        self.distribution = CategoricalMasked(logits=action_logits, masks=masks)
        return self
