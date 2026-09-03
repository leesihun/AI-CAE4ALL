"""Decoder Module

Implements the hierarchical decoder network for the SimulGenVAE architecture.
The decoder progressively reconstructs simulation data from latent representations
using transposed convolutions and skip connections for better gradient flow.

Author: SiHun Lee, Ph.D.
Email: kevin1007kr@gmail.com
"""

import torch.nn as nn
from model.common import *
from torch.nn import functional as F
from model.losses import kl, kl_2
import torch

class UpsampleBlock(nn.Module):
    """Upsampling block using transposed convolution.
    
    Basic building block for decoder that increases spatial resolution
    while transforming channel dimensions.
    
    Args:
        in_channel (int): Number of input channels
        out_channel (int): Number of output channels
    """
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self._seq = nn.Sequential(
            nn.ConvTranspose1d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        """Forward pass through the upsampling block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, in_channel, sequence_length]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch, out_channel, sequence_length]
        """
        return self._seq(x)

class DecoderBlock(nn.Module):
    """Decoder block consisting of multiple UpsampleBlocks.
    
    Creates a sequence of upsampling blocks that progressively reconstruct
    features from compressed representations.
    
    Args:
        channels (list): List of channel dimensions for each layer transition
        small (bool): Whether to use small variant (currently unused in UpsampleBlock)
    
    Example:
        channels=[256, 128, 64] creates two UpsampleBlocks:
        - 256 -> 128 channels
        - 128 -> 64 channels
    """
    def __init__(self, channels, small):
        super().__init__()
        self.channels = channels
        modules = []

        for i in range(len(channels) - 1):
            modules.append(UpsampleBlock(channels[i], channels[i + 1]))
        self.module_list = nn.ModuleList(modules)

    def forward(self, x):
        """Forward pass through all UpsampleBlocks in sequence.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after passing through all UpsampleBlocks
        """
        for module in self.module_list:
            x = module(x)

        return x

class Decoder(nn.Module):
    """Main hierarchical decoder network for VAE.
    
    Implements a multi-scale decoder that reconstructs simulation data from latent
    representations. Uses skip connections and hierarchical latent conditioning
    for better reconstruction quality and gradient flow.
    
    Args:
        z_dim (int): Dimension of the main latent space
        hierarchical_dim (int): Dimension of hierarchical latent spaces
        num_filter_dec (list): Number of filters for each decoder layer
        num_node (int): Number of nodes in output simulation data
        num_time (int): Number of time steps in output data
        batch_size (int): Batch size for initialization
        small (bool): Whether to use smaller model variant
    
    Attributes:
        decoder_blocks (nn.ModuleList): List of decoder blocks for feature reconstruction
        decoder_residual_blocks (nn.ModuleList): Residual blocks for skip connections
        recon (nn.Sequential): Final reconstruction layer
        linear_layers (nn.ModuleList): Linear layers for latent space transformation
    """
    def __init__(self, z_dim, hierarchical_dim, num_filter_dec, num_node, num_time, batch_size, small):
        super().__init__()

        self.decoder_blocks = nn.ModuleList([])
        for i in range(len(num_filter_dec)-1):
            self.decoder_blocks.append(DecoderBlock([num_filter_dec[i], num_filter_dec[i+1]], small))

        self.decoder_residual_blocks = nn.ModuleList([])
        for i in range(len(num_filter_dec)-1):
            self.decoder_residual_blocks.append(DecoderResidualBlock(num_filter_dec[i+1], small))

        self.recon = nn.Sequential(
            nn.Conv1d(num_filter_dec[-1], num_node, kernel_size=1),
            nn.GroupNorm(group_norm_groups(num_node), num_node),
            nn.Tanh()
        )

        self.zs = []
        self.num_filter_dec = num_filter_dec
        self.num_time = num_time

        latent_dim = z_dim
        hierarchical_dim = hierarchical_dim
        batch_size = batch_size

        self.sequence_start = nn.ModuleList([])
        self.sequence_start.append(nn.Sequential(
            nn.Linear(latent_dim, latent_dim*num_time),
            nn.Unflatten(1, (latent_dim, num_time)),
            nn.Conv1d((latent_dim), self.num_filter_dec[0], kernel_size=5, padding=2),
            nn.GroupNorm(group_norm_groups(self.num_filter_dec[0]), self.num_filter_dec[0]),
            nn.GELU(),
        ))

        self.xs_sequence = nn.ModuleList([])
        for i in range(len(num_filter_dec)-1):
            self.xs_sequence.append(nn.Sequential(
                nn.Linear(hierarchical_dim, hierarchical_dim*num_time),
                nn.Unflatten(1, (hierarchical_dim, num_time)),
                nn.Conv1d(hierarchical_dim, self.num_filter_dec[i+1], kernel_size=5, padding=2),
                nn.GroupNorm(group_norm_groups(self.num_filter_dec[i+1]), self.num_filter_dec[i+1]),
                nn.GELU(),
            ))

        self.condition_z = nn.ModuleList([])
        for i in range(len(num_filter_dec)-1):
            self.condition_z.append(nn.Sequential(
                ResidualBlock(num_filter_dec[i+1], small),
                nn.GELU(),
                nn.Conv1d(num_filter_dec[i+1], 2*num_filter_dec[i+1], kernel_size=3, padding=1),
                # nn.BatchNorm1d(2*num_filter_dec[i+1]),
            ))

        self.condition_xz = nn.ModuleList([])
        for i in range(len(num_filter_dec)-1):
            self.condition_xz.append(nn.Sequential(
                ResidualBlock(2*num_filter_dec[i+1], small),
                nn.GELU(),
                nn.Conv1d(2*num_filter_dec[i+1], 2*num_filter_dec[i+1], kernel_size=3, padding=1),
                # nn.BatchNorm1d(2*num_filter_dec[i+1]),
            ))

        self.small = small

    def forward(self, z, xs=None, mode = "random", freeze_level = -1):

        kl_losses = []

        for i in range(len(self.decoder_residual_blocks)):
            if i==0:
                z_sample = self.sequence_start[0](z)

            else:
                z_sample = torch.add(decoder_out, z)

            decoder_out = self.decoder_blocks[i](z_sample)
            decoder_out = self.decoder_residual_blocks[i](decoder_out)

            if i==len(self.decoder_residual_blocks)-1:
                break

            mu, log_var = self.condition_z[i](decoder_out).chunk(2, dim=1)

            if xs is not None:
                xs_sample = self.xs_sequence[i](xs[i])

                delta_mu, delta_log_var = self.condition_xz[i](torch.cat([xs_sample, decoder_out], dim=1)).chunk(2, dim=1)
                kl_losses.append(kl_2(delta_mu, delta_log_var, mu, log_var))

                mu = mu + delta_mu
                log_var = log_var + delta_log_var

                # Clamp log_var for numerical stability before computing std
                log_var = torch.clamp(log_var, min=-30, max=30)
                std = torch.exp(0.5*log_var)
                
                if mode=="fix" and i<freeze_level:
                    if len(self.zs) < freeze_level+1:
                        z = reparameterize(mu, std*1e-10)
                        self.zs.append(z)
                    else:
                        z = self.zs[i+1]

                elif mode== "fix":
                    z= reparameterize(mu, std*1e-10)
                else:
                    z=reparameterize(mu, std)

        x_hat = self.recon(decoder_out)

        return x_hat, kl_losses

def reparameterize(mu, std):
    # Clamp std to prevent very small values that could cause numerical issues
    std = torch.clamp(std, min=1e-8, max=10.0)
    eps = torch.randn_like(std)
    z = mu + eps*std
    return z