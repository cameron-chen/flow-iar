from .argmax_product import BinaryProductArgmaxSurjection
from .bijections.coupling import (AffineCouplingBijection,
                                  GaussianMixtureCouplingBijection,
                                  LogisticMixtureCouplingBijection)
from .bijections.thresholding import Threshold
from .cond_argmax import ConditionalDiscreteArgmaxSurjection
from .cond_argmax_product import ConditionalBinaryProductArgmaxSurjection
from .residual import Residual_V2 as Residual
from .squeeze1d import Squeeze1d
from .utils import base_to_integer, integer_to_base
