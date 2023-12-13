from normflows.flows import Residual
from survae.transforms import Transform

class Residual_V2(Residual, Transform): 
    def __init__(self, net, n_exact_terms=2, n_samples=1, reduce_memory=True, reverse=True):
        Residual.__init__(self, net, n_exact_terms, n_samples, reduce_memory, reverse)

    def inverse(self, z):
        z_out, _ = Residual.inverse(self, z)
        return z_out