"""Encoder Module

Implements the hierarchical encoder network for the SimulGenVAE architecture.
The encoder progressively compresses simulation data into hierarchical latent representations
using 1D convolutional blocks with GroupNorm for batch-independent normalization.

Author: SiHun Lee, Ph.D.
Email: kevin1007kr@gmail.com
"""

import torch.nn as nn
from model.common import *

class ConvBlock(nn.Module):
    """Basic convolutional block for encoder layers.
    
    Implements either a simple (small=True) or complex (small=False) convolutional
    block with GroupNorm normalization and GELU activation for stable training.
    
    Args:
        in_channel (int): Number of input channels
        out_channel (int): Number of output channels
        small (bool): If True, uses single conv layer; if False, uses two conv layers
    
    Note:
        GroupNorm is used instead of BatchNorm for better performance with
        varying batch sizes and distributed training.
    """
    def __init__(self, in_channel, out_channel, small):
        super().__init__()

        if small:
            self._seq = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0),
                nn.GroupNorm(group_norm_groups(out_channel), out_channel),
                nn.GELU(),
            )
        else:
            self._seq = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0),
                nn.GroupNorm(group_norm_groups(out_channel), out_channel),
                nn.GELU(),
                nn.Conv1d(out_channel, out_channel, kernel_size=3, padding=1),
                nn.GroupNorm(group_norm_groups(out_channel), out_channel),
                nn.GELU(),
            )
    
    def forward(self, x):
        """Forward pass through the convolutional block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, in_channel, sequence_length]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch, out_channel, sequence_length]
        """
        return self._seq(x)

class EncoderBlock(nn.Module):
    """Encoder block consisting of multiple ConvBlocks.
    
    Creates a sequence of convolutional blocks that progressively transform
    the input through different channel dimensions.
    
    Args:
        channels (list): List of channel dimensions for each layer transition
        small (bool): Whether to use small variant of ConvBlocks
    
    Example:
        channels=[64, 128, 256] creates two ConvBlocks:
        - 64 -> 128 channels
        - 128 -> 256 channels
    """
    def __init__(self, channels, small):
        super().__init__()
        self.channels = channels
        modules = []

        for i in range(len(channels)-1):
            modules.append(ConvBlock(channels[i], channels[i+1], small))
        self.module_list = nn.ModuleList(modules)

    def forward(self, x):
        """Forward pass through all ConvBlocks in sequence.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after passing through all ConvBlocks
        """
        for module in self.module_list:
            x = module(x)
        return x

class Encoder(nn.Module):
    """Main hierarchical encoder network for VAE.
    
    Implements a multi-scale encoder that compresses simulation data into both
    hierarchical and main latent representations. Uses 1D convolutions to process
    temporal sequences and produces mean and log-variance for reparameterization.
    
    Args:
        z_dim (int): Dimension of the main latent space
        hierarchical_dim (int): Dimension of hierarchical latent spaces
        num_filter_enc (list): Number of filters for each encoder layer
        num_node (int): Number of nodes in input simulation data
        num_time (int): Number of time steps in input data
        small (bool): Whether to use smaller model variant
    
    Attributes:
        encoder_blocks (nn.ModuleList): List of encoder blocks for feature extraction
        mus (nn.ModuleList): Mean prediction layers for each hierarchical level
        log_vars (nn.ModuleList): Log-variance prediction layers for each level
    """
    def __init__(self, z_dim, hierarchical_dim, num_filter_enc, num_node, num_time, small):
        super().__init__()

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock([num_node, num_filter_enc[0]], small),
        ])

        for i in range(0, len(num_filter_enc)-1):
            self.encoder_blocks.append(EncoderBlock([num_filter_enc[i], num_filter_enc[i+1]], small))

        self.encoder_residual_blocks = nn.ModuleList([])

        for i in range(0, len(num_filter_enc)):
            if i==len(num_filter_enc):
                self.encoder_residual_blocks.append(EncoderResidualBlock(num_filter_enc[i], num_filter_enc[i], small))
            else:
                self.encoder_residual_blocks.append(EncoderResidualBlock(num_filter_enc[i], num_filter_enc[i], small))

        self.z_dim = z_dim
        self.num_filter_enc = num_filter_enc
        hierarchical_dim = hierarchical_dim

        self.xs_linear = nn.ModuleList([])
        for i in range(len(num_filter_enc)):
            self.xs_linear.append(nn.Linear(num_filter_enc[i]*num_time,int(hierarchical_dim)))

        self.last_x_linear = nn.Linear(num_filter_enc[-1]*num_time, 2*z_dim)

        self.small = small

    def forward(self, x):
        xs = []
        B, _, _ = x.shape
        i=0

        for encoder_block, residual_block in zip(self.encoder_blocks, self.encoder_residual_blocks):
            x=encoder_block(x)
            x=residual_block(x)
            last_x = x

            xs_reshaped = last_x.view(B, -1)
            xs_reshaped = self.xs_linear[i](xs_reshaped)
            xs.append(xs_reshaped)
            i=i+1

        last_x = last_x.view(B, -1)
        last_x = self.last_x_linear(last_x)
        
        mu = last_x[:, :self.z_dim]
        log_var = last_x[:, self.z_dim:]

        return mu, log_var, xs[:-1][::-1]