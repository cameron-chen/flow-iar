import numpy as np
import torch
from survae.distributions import ConditionalDistribution

from .cond_surjection import ConditionalSurjection
from .utils import base_to_integer, integer_to_base


class ConditionalBinaryProductArgmaxSurjection(ConditionalSurjection):
    '''
    A generative argmax surjection using a Cartesian product of binary spaces. Argmax is performed over the final dimension.

    Note: This is a conditional version of the original BinaryProductArgmaxSurjection.

    Args:
        encoder: ConditionalDistribution, a distribution q(z|x) with support over z s.t. x=argmax z.

    Example:
        Input tensor x of shape (B, D, L) with discrete values {0,1,...,C-1}:
        encoder should be a distribution of shape (B, D, L, D), where D=ceil(log2(C)).
        When e.g. C=27, we have D=5, such that 2**5=32 classes are represented.
    '''
    stochastic_forward = True

    def __init__(self, encoder, num_classes, noise_std=0.1):
        super(ConditionalBinaryProductArgmaxSurjection, self).__init__()
        assert isinstance(encoder, ConditionalDistribution)
        self.encoder = encoder
        self.num_classes = num_classes
        self.dims = self.classes2dims(num_classes)
        self.noise_std = noise_std
        self.noise_eps = None
        self.explore_mode = False

    @staticmethod
    def classes2dims(num_classes):
        return int(np.ceil(np.log2(num_classes)))

    def idx2base(self, idx_tensor):
        return integer_to_base(idx_tensor, base=2, dims=self.dims)

    def base2idx(self, base_tensor):
        return base_to_integer(base_tensor, base=2)

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
        #   in what cases do we want to add noise?
        #   - interacting with the environment
        #   - [NOT] training the model
        #   - [NOT] evaluating the actions
        if self.explore_mode and (eps > 0.0): 
            noise = torch.stack([
                (std * torch.randn_like(el)) if (eps>np.random.rand()) 
                else torch.zeros_like(el) for el in z], dim=0)
            z = z + noise

        # inverse transform does not require context as z is conditional on context already.
        binary = torch.gt(z, 0.0).long()
        idx = self.base2idx(binary)
        return idx

    def turn_on_explore_mode(self):
        self.explore_mode = True
        assert self.noise_eps is not None, "noise_eps must be set before turning on explore mode."
    
    def turn_off_explore_mode(self):
        self.explore_mode = False

    def set_noise_eps(self, noise_eps: float):
        assert (noise_eps >= 0) and (noise_eps <= 1)
        self.noise_eps = noise_eps