"""VAE Network Module for SimulGenVAE

This module implements the core Variational Autoencoder architecture featuring:
- Hierarchical encoder-decoder networks for multi-scale representation
- Multiple loss functions (MSE, MAE, SmoothL1, Huber)
- Advanced memory management and performance optimizations
- Comprehensive error handling for robust training

The VAE architecture is specifically designed for physics simulation data with
temporal and spatial dependencies, supporting both single and batch processing.

Author: SiHun Lee, Ph.D.
Contact: kevin1007kr@gmail.com
Version: 2.0.0 (Refactored)
"""

# Core PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any

# SimulGenVAE modules
from model.encoder import Encoder
from model.decoder import Decoder, reparameterize
from model.losses import kl
from model.common import add_sn

class VAE(nn.Module):
    """Main Variational Autoencoder class for simulation data processing.
    
    This VAE implements a hierarchical latent space architecture with encoder-decoder
    networks optimized for transient and static simulation data. Features include
    multiple loss functions, gradient checkpointing support, and advanced memory management.
    
    Args:
        latent_dim (int): Main latent space dimension (typically 32)
        hierarchical_dim (int): Hierarchical latent dimension for multi-scale representation (typically 8)
        num_filter_enc (list): Number of filters for each encoder layer
        num_filter_dec (list): Number of filters for each decoder layer  
        num_node (int): Number of nodes in the simulation data
        num_time (int): Number of time steps in the simulation data
        lossfun (str): Loss function type. Options: 'MSE', 'MAE', 'smoothL1', 'Huber'
        batch_size (int): Batch size for training (used for decoder initialization)
        small (bool): Whether to use smaller model variant for memory efficiency
        use_checkpointing (bool): Enable gradient checkpointing (disabled for speed)
    
    Attributes:
        latent_dim (int): Dimension of the main latent space
        encoder (Encoder): Hierarchical encoder network
        decoder (Decoder): Hierarchical decoder network with skip connections
        lossfun (str): Selected loss function type
        loss_functions (dict): Pre-compiled loss functions for efficiency
        mse_loss (nn.MSELoss): MSE loss function (always available for metrics)
    """
    def __init__(self, latent_dim, hierarchical_dim, num_filter_enc, num_filter_dec, num_node, num_time, lossfun='MSE', batch_size=1, small=False, use_checkpointing=False):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, hierarchical_dim, num_filter_enc, num_node, num_time, small)
        self.decoder = Decoder(latent_dim, hierarchical_dim, num_filter_dec, num_node, num_time, batch_size, small)
        self.lossfun = lossfun
        # Checkpointing disabled for speed (user preference)
        self.use_checkpointing = False
        
        # Pre-compile loss functions for efficiency
        self.loss_functions = {
            'MSE': nn.MSELoss(),
            'MAE': nn.L1Loss(), 
            'smoothL1': nn.SmoothL1Loss(),
            'Huber': nn.HuberLoss()
        }
        self.mse_loss = nn.MSELoss()  # Always needed for recon_loss_MSE

    def forward(self, x):
        """Forward pass through the VAE.
        
        Performs encoding to latent space, reparameterization, and decoding back to
        reconstruction space. Includes comprehensive error handling for CUDA operations.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, num_node, num_time]
        
        Returns:
            tuple: (decoder_output, recon_loss, kl_losses, recon_loss_MSE)
                - decoder_output (torch.Tensor): Reconstructed data
                - recon_loss (torch.Tensor): Reconstruction loss using specified loss function
                - kl_losses (list): KL divergence losses [main_kl, hierarchical_kl_losses...]
                - recon_loss_MSE (torch.Tensor): MSE reconstruction loss for monitoring
        
        Raises:
            RuntimeError: If CUDA out of memory or other CUDA errors occur
        """
        try:
            # Always use regular forward pass - no speed trade-offs
            mu, log_var, xs = self.encoder(x)
            
            # Clamp log_var to prevent numerical instability before computing std
            log_var = torch.clamp(log_var, min=-30, max=30)
            std = torch.exp(0.5*log_var)
            z = reparameterize(mu, std)
            
            decoder_output, kl_losses = self.decoder(z, xs)

            # Use pre-compiled loss functions for efficiency
            recon_loss = self.loss_functions.get(self.lossfun, self.mse_loss)(decoder_output, x)
            recon_loss_MSE = self.mse_loss(decoder_output, x)

            kl_loss = kl(mu, log_var)

            # Minimal cleanup - only delete large intermediate tensors
            del xs, z  # Keep mu, log_var for potential debugging
            return decoder_output, recon_loss, [kl_loss]+kl_losses, recon_loss_MSE
            
        except RuntimeError as e:
            print(f"Error in VAE forward pass: {e}")
            raise
    
    def compile_model(self, mode='max-autotune'):
        """Compile the model for better performance using torch.compile."""
        if mode is False or mode == 'none':
            return
            
        if not hasattr(torch, 'compile'):
            print("torch.compile not available")
            return
        
        # Clear any existing compilation cache
        try:
            torch._dynamo.reset()
        except:
            pass
            
        print(f"Compiling VAE model with mode '{mode}' for maximum performance...")
        try:
            # Use conservative compilation settings for complex models
            self.encoder = torch.compile(self.encoder, mode=mode, dynamic=True, fullgraph=False)
            self.decoder = torch.compile(self.decoder, mode=mode, dynamic=True, fullgraph=False)
            print("Model compilation complete - expect significant speedup after warmup")
        except Exception as e:
            print(f"Compilation failed ({e}), falling back to reduce-overhead mode...")
            try:
                self.encoder = torch.compile(self.encoder, mode='reduce-overhead')
                self.decoder = torch.compile(self.decoder, mode='reduce-overhead')
                print("Successfully compiled with reduce-overhead mode")
            except Exception as e2:
                print(f"All compilation failed ({e2}), running in eager mode")
                # Don't compile at all if everything fails
    
    def to(self, *args, **kwargs):
        """Override to method to optimize memory format for better performance."""
        device = args[0] if args else kwargs.get('device', None)
        if device:
            print("Moving model to device (channels_last disabled for compilation compatibility)")
            # Disable channels_last format change when using torch.compile
            # as it can cause JSON parsing errors in inductor backend
            self = super().to(*args, **kwargs)
            # self = self.to(memory_format=torch.channels_last)  # Disabled for compilation
            return self
        return super().to(*args, **kwargs)