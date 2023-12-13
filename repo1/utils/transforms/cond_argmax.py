import numpy as np
import torch
import torch.nn.functional as F
from survae.distributions import ConditionalDistribution

from .cond_surjection import ConditionalSurjection


class ConditionalDiscreteArgmaxSurjection(ConditionalSurjection):
    '''
    A generative argmax surjection using one-hot encoding. Argmax is performed over the final dimension.

    Note: This is a discrete version of the ConditionalBinaryProductArgmaxSurjection.

    Args:
        encoder: ConditionalDistribution, a distribution q(z|x) with support over z s.t. x=argmax z.

    Example:
        Input tensor x of shape (B, D, L) with discrete values {0,1,...,C-1}:
        encoder should be a distribution of shape (B, D, L, C).
    '''
    stochastic_forward = True

    def __init__(self, encoder, num_classes, noise_std=0.1):
        super(ConditionalDiscreteArgmaxSurjection, self).__init__()
        assert isinstance(encoder, ConditionalDistribution)
        self.encoder = encoder
        self.num_classes = num_classes
        self.noise_std = noise_std
        self.noise_eps = None
        self.explore_mode = False

    def forward(self, x, context):
        # Note: x is a discrete tensor, while context can be either discrete or continuous.

        # Transform
        z, log_qz = self.encoder.sample_with_log_prob(context_act=x, context_obs=context)
        ldj = -log_qz
        return z, ldj

    def inverse(self, z, context):
        eps = self.noise_eps
        std = self.noise_std

        # add noise in "noise_eps" proportion of the time,
        #   otherwise, do nothing
        # in what cases do we want to add noise?
        #   - interacting with the environment
        #   - [NOT] training the model
        #   - [NOT] evaluating the actions
        if self.explore_mode and (eps > 0): 
            noise = torch.stack([
                (std * torch.randn_like(el)) if (eps>np.random.rand()) 
                else torch.zeros_like(el) for el in z], dim=0)
            z = z + noise

        # inverse transform does not require context as z is conditional on context already.
        idx = torch.argmax(z, dim=-1)
        return idx

    def inverse_soft(self, z, context):
        # inverse transform does not require context as z is conditional on context already.
        z_soft = F.softmax(z, dim=-1)
        return z_soft

    def turn_on_explore_mode(self):
        self.explore_mode = True
        assert self.noise_eps is not None, "noise_eps must be set before turning on explore mode."
    
    def turn_off_explore_mode(self):
        self.explore_mode = False

    def set_noise_eps(self, noise_eps):
        assert (noise_eps >= 0) and (noise_eps <= 1)
        self.noise_eps = noise_eps