import torch as th
from survae.distributions import ConditionalDistribution
from survae.transforms import Softplus

from ..transforms.utils import integer_to_base


class ObsCondBinaryEncoder(ConditionalDistribution):
    '''An encoder for BinaryProductArgmaxSurjection with a discrete input (action)
    and a continous input (observation).'''

    def __init__(self, noise_dist, dims):
        super(ObsCondBinaryEncoder, self).__init__()
        self.noise_dist = noise_dist
        self.dims = dims
        self.softplus = Softplus()

    def sample_with_log_prob(self, context_act, context_obs):
        # Example: context_act.shape = (B, C, H, W) with values in {0,1,...,K-1}
        # Sample z.shape = (B, C, H, W, K)

        binary = integer_to_base(context_act, base=2, dims=self.dims)
        sign = binary * 2 - 1

        u, log_pu = self.noise_dist.sample_with_log_prob(context=[context_act, context_obs])
        u_positive, ldj = self.softplus(u)

        log_pu_positive = log_pu - ldj
        z = u_positive * sign

        log_pz = log_pu_positive
        return z, log_pz

    @th.no_grad()
    def sample(self, context_act, context_obs):
        # Example: context_act.shape = (B, C, H, W) with values in {0,1,...,K-1}
        # Sample z.shape = (B, C, H, W, K)

        binary = integer_to_base(context_act, base=2, dims=self.dims)
        sign = binary * 2 - 1

        u, _ = self.noise_dist.sample_with_log_prob(context=[context_act, context_obs])
        u_positive, _ = self.softplus(u)

        z = u_positive * sign

        return z

    def sample_enable_grad(self, context_act, context_obs):
        # Example: context_act.shape = (B, C, H, W) with values in {0,1,...,K-1}
        # Sample z.shape = (B, C, H, W, K)

        binary = integer_to_base(context_act, base=2, dims=self.dims)
        sign = binary * 2 - 1

        u, _ = self.noise_dist.sample_with_log_prob(context=[context_act, context_obs])
        u_positive, _ = self.softplus(u)

        z = u_positive * sign

        return z