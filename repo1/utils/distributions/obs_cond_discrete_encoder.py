import torch as th
import torch.nn.functional as F
from survae.distributions import ConditionalDistribution

from utils.transforms import Threshold


class ObsCondDiscreteEncoder(ConditionalDistribution):
    '''An encoder for DiscreteArgmaxSurjection with a discrete input (action)
    and a continous input (observation).'''

    def __init__(self, noise_dist, dims):
        super(ObsCondDiscreteEncoder, self).__init__()
        self.noise_dist = noise_dist
        self.num_classes = dims
        self.threshold = Threshold()

    def sample_with_log_prob(self, context_act, context_obs):
        # Example: context_act.shape = (B, C, H, W) with values in {0,1,...,K-1}
        # Sample z.shape = (B, C, H, W, K)

        one_hot = F.one_hot(context_act, num_classes=self.num_classes)

        u, log_pu = self.noise_dist.sample_with_log_prob(context=[context_act, context_obs])
        z, ldj = self.threshold(u, one_hot)
        log_pz = log_pu - ldj

        return z, log_pz

    @th.no_grad()
    def sample(self, context_act, context_obs):
        # Example: context_act.shape = (B, C, H, W) with values in {0,1,...,K-1}
        # Sample z.shape = (B, C, H, W, K)

        one_hot = F.one_hot(context_act, num_classes=self.num_classes)

        u, _ = self.noise_dist.sample_with_log_prob(context=[context_act, context_obs])
        z, _ = self.threshold(u, one_hot)

        return z

    def sample_enable_grad(self, context_act, context_obs):
        # Example: context_act.shape = (B, C, H, W) with values in {0,1,...,K-1}
        # Sample z.shape = (B, C, H, W, K)

        one_hot = F.one_hot(context_act, num_classes=self.num_classes)

        u, _ = self.noise_dist.sample_with_log_prob(context=[context_act, context_obs])
        z, _ = self.threshold(u, one_hot)

        return z