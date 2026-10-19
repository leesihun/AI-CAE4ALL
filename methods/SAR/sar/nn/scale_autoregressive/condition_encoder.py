import torch
from torch import nn

from ..models.vae import DownBlock
from ..modules.transolver import Transolver
from ...graph import Graph


class ConditionEncoder(nn.Module):
    """
    Component 1 of SAR: Condition Encoder for Global Context Embedding
    
    The Condition Encoder processes the entire node set to embed geometric and physical 
    information into node-wise feature vectors that capture both local features and 
    global context across the domain. It implements:
    
    Y = ConditionEncoder(X, V_c, Γ)
    
    where:
    - X: spatial coordinates of all nodes
    - V_c: conditioning attributes (physical parameters, boundary conditions, etc.)
    - Γ: scale assignment for each node
    - Y: output conditioning vectors with global context
    
    Key characteristics:
    - Evaluated once per sample (can be cached for multiple samples)
    - Operates on full node set V to provide global context
    - Lightweight compared to autoregressive module and sampler
    - Handles multi-scale information through down-sampling blocks
    - Uses Transolver for efficient global interaction modeling
    
    Architecture:
    1. Input processing for conditioning features
    2. Multi-scale down-sampling (if ref_scale > 1) 
    3. Global context aggregation via Transolver
    4. Output conditioning vectors Y for each node
    """
    
    def __init__(
        self,
        dim:           int,                    # Spatial dimension (2D or 3D)
        cond_features: int,                    # Conditioning feature dimension
        num_iters:     int,                    # Number of autoregressive steps (scales)
        out_features:  int,                    # Output conditioning dimension
        ref_scale:     int,                    # Reference scale for multi-scale processing
        d_model:       int,                    # Hidden dimension
        num_layers:    int,                    # Number of Transolver layers
        num_heads:     int,                    # Number of attention heads
        num_slices:    int,                    # Number of slices for physics-informed attention
        dropout:       float,                  # Dropout rate
    ) -> None:
        """
        Initialize condition encoder for SAR.
        
        The condition encoder creates global conditioning vectors that ensure each node
        retains access to full geometric and physical context during autoregressive
        generation, even when finer-scale information is not yet available.
        
        Args:
            dim: Spatial dimension of the problem
            cond_features: Dimension of conditioning attributes (physics parameters, etc.)
            num_iters: Number of autoregressive steps (determines scale encoding dimension)
            out_features: Dimension of output conditioning vectors
            ref_scale: Reference scale for multi-scale down-sampling
            d_model: Hidden dimension for internal processing
            num_layers: Depth of Transolver for global context aggregation
        """
        super().__init__()
        
        # Store hyperparameters
        self.dim           = dim
        self.cond_features = cond_features
        self.num_iters     = num_iters
        self.out_features  = out_features
        self.ref_scale     = ref_scale
        self.d_model       = d_model
        self.num_layers    = num_layers
        self.num_heads     = num_heads
        self.num_slices    = num_slices
        self.dropout       = dropout
        
        # Input layer for conditioning features
        # Projects conditioning attributes to hidden dimension if multi-scale processing needed
        self.in_layer = nn.Linear(self.cond_features, self.d_model) if self.ref_scale > 1 else nn.Identity()
        
        # Multi-scale down-sampling blocks
        # Used when working with VAE (ref_scale > 1) to process hierarchical information
        # Each down block aggregates information from finer to coarser scales
        self.down_blocks = nn.ModuleList([
            DownBlock(
                scale          = l,                    # Current scale level
                depth          = n,                    # Number of message passing layers
                fnns_depth     = 2,                    # Depth of MLPs in each layer
                fnns_width     = self.d_model,         # Width of MLPs
                activation     = nn.SELU,              # Activation function
                dropout        = self.dropout,         # Dropout rate
                aggr           = 'sum',                # Aggregation method for message passing
                dim            = self.dim,             # Spatial dimension
            ) for l, n in enumerate([2,] * (self.ref_scale - 1))  # 2 layers per scale
        ])
        
        # Middle layer for processing down-sampled features
        # Transforms multi-scale features back to conditioning dimension
        self.mid_layer = nn.Sequential(
            nn.SELU(),
            nn.Linear(self.d_model, self.cond_features),
        ) if self.ref_scale > 1 else nn.Identity()
        
        # Main Transolver for global context aggregation
        # Processes spatial coordinates + conditioning + scale information
        # to create globally-informed conditioning vectors
        self.transolver = Transolver(
            in_features  = self.dim + self.cond_features + self.num_iters,  # pos + conditioning + scale
            out_features = self.out_features,                               # Output conditioning dimension
            num_layers   = self.num_layers,                                 # Network depth
            d_model      = self.d_model,                                    # Hidden dimension
            num_heads    = self.num_heads,                                  # Attention heads
            num_slices   = self.num_slices,                                 # Physics-informed slices
            dropout      = self.dropout,                                    # Dropout rate
        )

    def forward(
        self,
        graph: Graph,              # Input graph with node/edge features and topology
        pos:   torch.Tensor,       # Spatial coordinates (num_nodes, dim)
        s:     torch.Tensor,       # Scale one-hot encoding (num_nodes, num_iters)
    ) -> torch.Tensor:
        """
        Forward pass: create global conditioning vectors.
        
        Processes the entire computational domain to create conditioning vectors Y 
        that capture both local geometric features and global physical context.
        These vectors ensure that during autoregressive generation, each node has
        access to full domain information even when finer scales are unknown.
        
        Args:
            graph: Input graph containing:
                   - node features (geometric, physical parameters)
                   - edge connectivity for message passing
                   - batch information for multiple graphs
            pos: Spatial coordinates for each node
            s: One-hot encoding of scale assignment for each node
            
        Returns:
            Y: Global conditioning vectors for each node (num_nodes, out_features)
        """
        # Step 1: Process conditioning features
        # Combine available conditioning information (global params, local features, etc.)
        # Graph may contain: 'glob' (global params), 'loc' (local features), 'omega' (frequencies)
        conditioning_features = [
            f for f in [graph.get('glob'), graph.get('loc'), graph.get('omega')] 
            if f is not None
        ]
        
        # Project conditioning features to hidden dimension for multi-scale processing
        c = self.in_layer(
            torch.cat(conditioning_features, dim=1) if conditioning_features else 
            torch.zeros(pos.shape[0], self.cond_features, device=pos.device)
        ) 
        
        # Step 2: Multi-scale down-sampling (if using VAE with multiple scales)
        # Hierarchically aggregate information from fine to coarse scales
        # This captures multi-resolution geometric and physical context
        for down_block in self.down_blocks:
            c = down_block(graph, c)
        
        # Step 3: Transform multi-scale features back to conditioning dimension
        c = self.mid_layer(c)
        
        # Step 4: Global context aggregation via Transolver
        # Combine spatial coordinates, processed conditioning, and scale information
        # The Transolver creates globally-informed representations through attention
        y = self.transolver(
            torch.cat([
                pos,    # Spatial coordinates (geometric context)
                c,      # Processed conditioning features (physical context)  
                s,      # Scale one-hot encoding (hierarchical context)
            ], dim=1),
            graph.batch,  # Batch assignment for multiple graphs
        )
        
        return y