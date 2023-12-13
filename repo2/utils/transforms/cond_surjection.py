from survae.transforms.cond_base import ConditionalTransform


class ConditionalSurjection(ConditionalTransform):
    """Base class for Conditional Surjection"""

    bijective = False

    @property
    def stochastic_forward(self):
        raise NotImplementedError()

    @property
    def stochastic_inverse(self):
        return not self.stochastic_forward
