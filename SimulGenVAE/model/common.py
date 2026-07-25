"""Common Utilities Module

Provides shared utilities for the SimulGenVAE architecture including
weight initialization, spectral normalization, and activation functions.

Author: SiHun Lee, Ph.D.
Email: kevin1007kr@gmail.com
"""

import torch.nn as nn
import torch
import numpy as np
from torch.nn.utils import spectral_norm

def add_sn(m):
    """Apply spectral normalization to a module.
    
    Adds spectral normalization to convolutional and linear layers to
    constrain the Lipschitz constant for training stability.
    
    Args:
        m (nn.Module): Module to apply spectral normalization to
    
    Returns:
        nn.Module: Module with spectral normalization applied (if applicable)
    
    Note:
        Only applies to Conv1d, ConvTranspose1d, Conv2d, ConvTranspose2d, and Linear layers.
    """
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        if m.weight.numel() > 0:
            return spectral_norm(m)
        else:
            print(f'Warning: Cannot apply spectral normalization to {type(m).__name__} - weight tensor is empty')
            return m
    else:
        return m

def initialize_weights_He(m):
    """Initialize module weights using He (Kaiming) initialization.
    
    Applies Kaiming uniform initialization to convolutional and linear layers.
    This initialization is particularly effective for ReLU-based networks.
    
    Args:
        m (nn.Module): Module to initialize
    
    Note:
        - Conv layers: Uses Kaiming uniform with 'relu' nonlinearity
        - Linear layers: Uses standard Kaiming uniform
        - Biases are initialized to zero
    """
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d,nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
    
import math

class Swish(nn.Module):
    """Swish activation function.
    
    Implements Swish activation: f(x) = x * sigmoid(x)
    Provides smooth, non-monotonic activation that can outperform ReLU.
    
    Reference:
        Searching for Activation Functions (Ramachandran et al., 2017)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x*torch.sigmoid(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, dim, small):
        super().__init__()

        if small:
            self._seq = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3, padding=1),
                nn.GroupNorm(min(8, max(1, dim//4)), dim),
                # nn.LayerNorm(dim), 
                nn.GELU(),
            )
        else:
            self._seq = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3, padding=1),
                nn.GroupNorm(min(8, max(1, dim//4)), dim),
                # nn.LayerNorm(dim), 
                nn.GELU(),
                nn.Conv1d(dim, dim, kernel_size=3, padding=1),
                nn.GroupNorm(min(8, max(1, dim//4)), dim),
                # nn.LayerNorm(dim), 
                nn.GELU(),
            )

    def forward(self, x):
        return x + 0.1*self._seq(x)
    
class EncoderResidualBlock(nn.Module):
    def __init__(self, input, dim, small):
        super().__init__()

        if small:
            self.seq = nn.Sequential(
                nn.Conv1d(input, input, kernel_size=3, padding=1),
                nn.GroupNorm(min(8, max(1, input//4)), input),
                nn.GELU(),
            )
        else:
            self.seq = nn.Sequential(
                nn.Conv1d(input, input, kernel_size=3, padding=1),
                nn.GroupNorm(min(8, max(1, input//4)), input),
                nn.GELU(),
                nn.Conv1d(input, input, kernel_size=3, padding=1),
                nn.GroupNorm(min(8, max(1, input//4)), input),
                nn.GELU(),
            )

    def forward(self, x):
        return x+0.1*self.seq(x)
    
class DecoderResidualBlock(nn.Module):
    def __init__(self, input, small):
        super().__init__()
        EXPANSION_MULTIPLE = 5  # Channel expansion factor for decoder residual blocks
        multiple = EXPANSION_MULTIPLE

        if small:
            self.seq = nn.Sequential(
                nn.Conv1d(input, input*multiple, kernel_size=1),
                nn.GroupNorm(min(8, max(1, (input*multiple)//4)), input*multiple),
                nn.GELU(),
                nn.Conv1d(input*multiple, input*multiple, kernel_size=5, padding=2),
                nn.GroupNorm(min(8, max(1, (input*multiple)//4)), input*multiple),
                nn.GELU(),
                nn.Conv1d(input*multiple, input, kernel_size=1, padding=0),
                nn.GroupNorm(min(8, max(1, input//4)), input),
                nn.GELU(),
            )
        else:
            self.seq = nn.Sequential(
                nn.Conv1d(input, input, kernel_size=1),
                nn.GroupNorm(min(8, max(1, input//4)), input),
                nn.GELU(),
                nn.Conv1d(input, input*multiple, kernel_size=5, padding=2),
                nn.GroupNorm(min(8, max(1, (input*multiple)//4)), input*multiple),
                nn.GELU(),
                nn.Conv1d(input*multiple, input*multiple, kernel_size=5, padding=2),
                nn.GroupNorm(min(8, max(1, (input*multiple)//4)), input*multiple),
                nn.GELU(),
                nn.Conv1d(input*multiple, input, kernel_size=1, padding=0),
                nn.GroupNorm(min(8, max(1, input//4)), input),
                nn.GELU(),
            )

    def forward(self, x):
        return x+0.1*self.seq(x)