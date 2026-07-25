"""Enhanced Parametric Latent Conditioner Model

Implements MLP-based latent conditioning for parametric/tabular data in SimulGenVAE.
This model processes numerical features to predict latent space representations
for both main and hierarchical latent variables.

Enhanced Features:
- Multi-layer perceptron with progressive dropout and residual connections
- Layer normalization for training stability (batch-independent)
- Adaptive bottleneck sizing based on input complexity
- Dual output heads with shared feature extraction
- GELU activations with proper gradient flow
- Gradient scaling and numerical stability improvements

Author: SiHun Lee, Ph.D.
Email: kevin1007kr@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow in deep networks."""
    
    def __init__(self, input_dim, output_dim, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Main path
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.ln1 = nn.LayerNorm(output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.ln2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Skip connection (if dimensions don't match)
        self.skip_connection = None
        if input_dim != output_dim:
            self.skip_connection = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
    
    def forward(self, x):
        identity = x
        
        # Main path
        out = self.linear1(x)
        out = self.ln1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.ln2(out)
        
        # Skip connection
        if self.skip_connection is not None:
            identity = self.skip_connection(x)
        
        out += identity
        out = F.gelu(out)
        
        return out


class LatentConditioner(nn.Module):
    """Enhanced MLP-based latent conditioner for parametric data.
    
    Processes tabular/parametric input data to predict latent space representations.
    Uses residual connections, batch normalization, and adaptive architecture
    for improved gradient flow and training stability.
    
    Args:
        latent_conditioner_filter (list): Number of neurons in each hidden layer
        latent_dim_end (int): Dimension of main latent space (typically 32)
        input_shape (int): Number of input features/parameters
        latent_dim (int): Dimension of hierarchical latent space (typically 8)
        size2 (int): Size multiplier for hierarchical latent output
        dropout_rate (float): Base dropout rate for regularization (default: 0.3)
    
    Attributes:
        input_norm (nn.LayerNorm): Input normalization layer
        backbone (nn.ModuleList): Residual blocks for feature extraction
        feature_projection (nn.Sequential): Final feature projection layer
        latent_out (nn.Sequential): Output head for main latent predictions
        xs_out (nn.Sequential): Output head for hierarchical latent predictions
    
    Enhancements:
        - Residual connections for better gradient flow
        - Layer normalization for training stability (batch-independent)
        - Progressive dropout (lighter early, heavier late)
        - Adaptive bottleneck sizing based on input complexity
        - Proper weight initialization for faster convergence
    """
    def __init__(self, latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, dropout_rate=0.3):
        super(LatentConditioner, self).__init__()
        
        self.latent_dim = latent_dim
        self.size2 = size2
        self.latent_conditioner_filter = latent_conditioner_filter
        self.latent_dim_end = latent_dim_end
        self.input_shape = input_shape
        self.num_latent_conditioner_filter = len(self.latent_conditioner_filter)
        self.dropout_rate = dropout_rate

        # Input normalization for numerical stability
        self.input_norm = nn.LayerNorm(input_shape)
        
        # Progressive dropout rates (lighter early layers, heavier later)
        dropout_schedule = [
            dropout_rate * 0.5,  # Early layers: lighter dropout
            dropout_rate * 0.7,
            dropout_rate * 1.0,  # Later layers: full dropout
            dropout_rate * 1.2
        ]
        
        # Backbone with residual connections
        self.backbone = nn.ModuleList()
        
        # First layer (input -> first hidden)
        current_dim = input_shape
        for i, next_dim in enumerate(self.latent_conditioner_filter):
            dropout_idx = min(i, len(dropout_schedule) - 1)
            current_dropout = dropout_schedule[dropout_idx]
            
            if i == 0:
                # First layer: simple linear + layer norm
                self.backbone.append(nn.Sequential(
                    nn.Linear(current_dim, next_dim),
                    nn.LayerNorm(next_dim),
                    nn.GELU(),
                    nn.Dropout(current_dropout)
                ))
            else:
                # Subsequent layers: residual blocks
                self.backbone.append(ResidualBlock(current_dim, next_dim, current_dropout))
            
            current_dim = next_dim
        
        # Final feature projection
        final_feature_size = self.latent_conditioner_filter[-1]
        
        # Adaptive bottleneck sizing based on problem complexity
        # For parametric data with 484 features -> more conservative bottleneck
        complexity_ratio = min(8, max(2, input_shape // 64))  # Adaptive ratio based on input size
        hidden_size = max(self.latent_dim_end * 2, final_feature_size // complexity_ratio)
        
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(final_feature_size),
            nn.Dropout(dropout_rate * 0.8),  # Moderate dropout before output heads
        )
        
        # Improved output heads with better capacity
        self.latent_out = nn.Sequential(
            nn.Linear(final_feature_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.6),  # Reduced dropout in output heads
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.4),
            nn.Linear(hidden_size // 2, self.latent_dim_end),
            nn.Tanh()
        )
        
        self.xs_out = nn.Sequential(
            nn.Linear(final_feature_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.4),
            nn.Linear(hidden_size // 2, self.latent_dim * self.size2),
            nn.Tanh()
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Proper weight initialization for better gradient flow."""
        if isinstance(module, nn.Linear):
            # Xavier/Glorot initialization for GELU activations
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Input normalization
        x = self.input_norm(x)
        
        # Forward through backbone layers
        for layer in self.backbone:
            x = layer(x)
        
        # Feature projection
        features = self.feature_projection(x)
        
        # Dual output heads
        latent_out = self.latent_out(features)
        xs_out = self.xs_out(features)
        xs_out = xs_out.view(xs_out.size(0), self.size2, self.latent_dim)

        return latent_out, xs_out