from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def kl(mu, log_var):
    """Compute KL divergence loss for VAE latent variables.
    
    Calculates KL divergence between learned latent distribution q(z|x) and
    standard Gaussian prior p(z) = N(0, I). Includes numerical stability
    measures to prevent overflow/underflow.
    
    Args:
        mu (torch.Tensor): Mean of latent distribution, shape [batch, latent_dim]
        log_var (torch.Tensor): Log variance of latent distribution, same shape as mu
    
    Returns:
        torch.Tensor: Scalar KL divergence loss
    
    Note:
        Uses clamping on log_var to range [-30, 30] for numerical stability.
        This corresponds to standard deviations from ~6e-14 to ~2.6e13.
    """
    # Clamp log_var to prevent numerical instability
    # Start with wider range [-30, 30], narrow to [-20, 20] only if needed
    # This covers std from 6e-14 to 2.6e13 - should be sufficient for most cases
    log_var = torch.clamp(log_var, min=-30, max=30)
    var = torch.exp(log_var)
    loss = 0.5*torch.sum(mu**2+var-log_var-1, dim=[1])
    return torch.mean(loss, dim=0)

def kl_2(delta_mu, delta_log_var, mu, log_var):
    # Clamp log_var values to prevent numerical instability
    # Start with wider range, tighten if you get NaN/Inf
    log_var = torch.clamp(log_var, min=-30, max=30)
    delta_log_var = torch.clamp(delta_log_var, min=-30, max=30)
    
    var = torch.exp(log_var)
    delta_var = torch.exp(delta_log_var)
    
    # Add small epsilon to prevent division by zero
    eps = 1e-8
    var = var + eps
    
    loss = 0.5*torch.sum(torch.div(delta_var, var)+torch.div((mu-delta_mu)**2, var)-delta_log_var+log_var-1, dim=[1,2])
    return torch.mean(loss, dim=0)

def log_sum_exp(x):
    m2 = torch.max(x, dim=1, keepdim=True)[0]
    m=m2.unsqueeze(1)
    return m+torch.log(torch.sum(torch.exp(x-m2), dim=1))
