import math


def loglik_bpd_cond(model, x, context):
    """Compute the log-likelihood in bits per dim for conditional flow."""
    return - model.log_prob(x, context).sum() / (math.log(2) * x.shape.numel())

def elbo_bpd_cond(model, x, context):
    """
    Compute the ELBO in bits per dim for conditional flow.
    Same as .loglik_bpd_cond(), but may improve readability.
    """
    return loglik_bpd_cond(model, x, context)
