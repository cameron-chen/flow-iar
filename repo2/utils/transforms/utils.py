import torch


def integer_to_base(idx_tensor, base, dims):
    '''
    Encodes index tensor to a Cartesian product representation.
    Args:
        idx_tensor (LongTensor): An index tensor, shape (...), to be encoded.
        base (int): The base to use for encoding.
        dims (int): The number of dimensions to use for encoding.
    Returns:
        LongTensor: The encoded tensor, shape (..., dims).
    '''
    powers = base ** torch.arange(dims - 1, -1, -1, device=idx_tensor.device)
    # idx_tensor is drawn from a categorical distribution (so no negative numbers), 
    #    so either rounding mode "floor" or "trunc" will work.
    floored = torch.div(idx_tensor[..., None], powers, rounding_mode='floor')
    remainder = floored % base

    base_tensor = remainder
    return base_tensor


def base_to_integer(base_tensor, base):
    '''
    Decodes Cartesian product representation to an index tensor.
    Args:
        base_tensor (LongTensor): The encoded tensor, shape (..., dims).
        base (int): The base used in the encoding.
    Returns:
        LongTensor: The index tensor, shape (...).
    '''
    dims = base_tensor.shape[-1]
    powers = base ** torch.arange(dims - 1, -1, -1, device=base_tensor.device)
    powers = powers[(None,) * (base_tensor.dim()-1)]

    idx_tensor = (base_tensor * powers).sum(-1)
    return idx_tensor
