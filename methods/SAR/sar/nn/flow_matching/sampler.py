import torch
from torch import nn
from torch_geometric.utils import scatter

from . import FlowMatchingModel
from ..modules.sin_embedding import SinusoidalTimeEmbedding
from ..modules.transolver import Transolver
from ..modules.resnet import ResNet


class Sampler(FlowMatchingModel):
    """
    Component 3 of SAR: Flow Matching Sampler for Physical Field Generation
    
    The Sampler is the probabilistic component of the SAR model responsible for generating 
    field values at each autoregressive step. It implements the conditional distribution:
    S_k ~ Sampler(S_k | X_k, Y_k, Z_k, Γ_k)
    
    where:
    - S_k: field values at scale k
    - X_k: spatial coordinates at scale k  
    - Y_k: global conditioning from condition encoder
    - Z_k: autoregressive conditioning from AR module
    - Γ_k: scale information
    
    Uses flow matching to transform Gaussian noise to target distribution via learned ODE.
    The key insight is adaptive step allocation: more denoising steps for coarser scales
    (higher uncertainty), fewer steps for finer scales (benefit from strong conditioning).
    
    Architecture: 
    - Time embedding for denoising step r ∈ [0,1]
    - Input processing combining state, conditioning, and time
    - Main model (Transolver or ResNet) with time conditioning
    - Output projection to velocity field
    """
    
    def __init__(
        self,
        num_features:  int,                    # Dimension of physical fields to generate
        cond_features: int,                    # Dimension of conditioning (Y_k + Z_k)
        d_model:       int,                    # Hidden dimension
        d_r:           int,                    # Time embedding dimension
        num_layers:    int,                    # Number of layers in main model
        num_heads:     int   = None,           # Attention heads (for Transolver)
        num_slices:    int   = None,           # Physics-informed attention slices (for Transolver)
        dropout:       float = 0.0,            # Dropout rate
        arch:          str   = 'transolver',   # Architecture: 'transolver' or 'resnet'
    ) -> None:
        """
        Initialize flow matching sampler for SAR.
        
        The sampler learns to generate field values by modeling the velocity field
        of a continuous normalizing flow that transforms Gaussian noise q_{k,0} 
        to the target distribution q_{k,1} via the ODE: dx/dr = u_r(x_r | conditioning)
        
        Args:
            num_features: Dimension of physical fields (e.g., velocity components + pressure)
            cond_features: Combined conditioning dimension (from Y_k and Z_k)
            d_model: Hidden dimension for internal processing
            d_r: Dimension of time embedding for denoising step r
            num_layers: Depth of main model
            arch: Architecture choice - 'transolver' for attention-based or 'resnet' for MLP-based
        """
        super().__init__()
        
        # Store hyperparameters
        self.num_features  = num_features
        self.cond_features = cond_features
        self.d_model       = d_model
        self.d_r           = d_r
        self.num_layers    = num_layers
        self.num_heads     = num_heads
        self.num_slices    = num_slices
        self.dropout       = dropout
        self.arch          = arch.lower()
        
        # Time embedding for denoising step r ∈ [0,1]
        # Maps continuous time to high-dimensional representation for conditioning
        self.r_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(self.d_model),  # Sinusoidal encoding like in Transformers
            nn.Linear(self.d_model, self.d_r),
            nn.SELU(),
        )
        
        # Input processing layer
        # Combines current state x_r with conditioning (Y_k, Z_k) from SAR components
        self.in_layer  = nn.Sequential(
            nn.Linear(self.num_features + self.cond_features, 2 * self.d_model), 
            nn.SELU(),
            nn.Linear(2 * self.d_model, self.d_model),
        )
        
        # Time embedding projection for conditioning main model
        self.emb_layer = nn.Linear(self.d_r, self.d_model)
        
        # Main model: learns velocity field u_r(x_r | conditioning)
        # Choice of architecture affects computational efficiency vs. expressiveness
        if self.arch == 'transolver':
            # Attention-based: better for global interactions but more expensive
            # Feasible in SAR due to adaptive step allocation (fewer steps at fine scales)
            assert self.num_heads is not None, "num_heads must be specified for Transolver model"
            assert self.num_slices is not None, "num_slices must be specified for Transolver model"
            self.model = Transolver(
                in_features  = self.d_model + self.d_r,
                out_features = self.d_model,
                num_layers   = self.num_layers,
                d_model      = self.d_model,
                num_heads    = self.num_heads,
                num_slices   = self.num_slices,
                d_cond       = self.d_r,            # Conditioned on time embedding
                dropout      = self.dropout,
            )
        elif self.arch == 'resnet':
            # MLP-based: more efficient but limited global interactions
            self.model = ResNet(
                in_features  = self.d_model + self.d_r,
                out_features = self.d_model,
                num_layers   = self.num_layers,
                d_model      = self.d_model,
                activation   = nn.SELU,
                d_cond       = self.d_r,            # Conditioned on time embedding
                dropout      = self.dropout,
            )
        else:
            raise ValueError(f"Unknown model type: {self.arch}. Choose 'transolver' or 'resnet'.")
        
        # Output layer: projects to velocity field dimension
        self.out_layer = nn.Linear(self.d_model, self.num_features)

    def forward(
        self,
        x:     torch.Tensor,     # Current state along flow trajectory (num_nodes, num_features)
        y:     torch.Tensor,     # Conditioning from SAR components (num_nodes, cond_features)  
        r:     torch.Tensor,     # Denoising time parameter ∈ [0,1] (batch_size,)
        batch: torch.LongTensor, # Batch assignment for nodes (num_nodes,)
    ) -> torch.Tensor:
        """
        Compute velocity field at denoising time r.
        
        Implements the velocity field u_r(x_r | Y_k, Z_k) in the flow matching ODE:
        dx/dr = u_r(x_r | Y_k, Z_k)
        
        where x_r evolves from Gaussian noise (r=0) to target distribution (r=1).
        
        Args:
            x: Current state tensor along flow trajectory
            y: Combined conditioning from SAR:
               - Y_k: spatial + global conditioning from condition encoder  
               - Z_k: autoregressive conditioning from AR module
            r: Time parameter tensor (0 = pure noise, 1 = target distribution)
            batch: Batch assignment for nodes
            
        Returns:
            Predicted velocity field directing flow toward target distribution
        """
        # Embed denoising time parameter into high-dimensional representation
        emb = self.r_embedding(r)
        
        # Process input by combining current state, SAR conditioning, and time
        # This creates a rich representation for velocity field prediction
        x = self.in_layer(torch.cat([x, y], dim=-1)) + self.emb_layer(emb)[batch]
        
        # Process through chosen architecture with time conditioning
        # The time embedding ensures the model knows where we are in the denoising process
        x = self.model(
            torch.cat([x, emb[batch]], dim=-1),  # Input + time conditioning
            batch,
            emb,                                 # Global time conditioning for attention/normalization
        )
        
        # Project to velocity field dimension (same as physical field dimension)
        return self.out_layer(x)
    
    def training_step(
        self,
        field_end: torch.Tensor,     # Target field values S_k (num_nodes, num_features)
        y:         torch.Tensor,     # SAR conditioning Y_k, Z_k (num_nodes, cond_features)
        batch:     torch.LongTensor, # Batch assignment (num_nodes,)
        repeat:    int = 1,          # Number of training samples per batch
    ) -> torch.Tensor:
        """
        Training step for flow matching objective.
        
        Implements the loss: L_k = E[||w_{k,r}(S_{k,r}) - u_{k,r}(S_{k,r} | Y_k, Z_k)||^2]
        where:
        - w_{k,r}: target velocity field (analytical from flow matching)
        - u_{k,r}: learned velocity field (this network)
        - S_{k,r}: intermediate state at time r
        
        Strategy: Sample random times r and states along trajectory, then learn
        to predict the correct velocity field at each point.
        """
        # Get batch size for sampling random time steps
        batch_size = batch.max().item() + 1
        
        # Sample random time steps r ∈ [0,1] for flow matching training
        # Each r corresponds to a different point along the denoising trajectory
        r = torch.rand(batch_size * repeat, device=self.device)
        
        # Repeat batch, conditioning, and targets for multiple training samples
        # This increases training diversity by sampling multiple r values per batch
        masked_batch = torch.cat([batch + i * batch_size for i in range(repeat)])
        y            = y.repeat(repeat, 1)
        field_end    = field_end.repeat(repeat, 1)
        
        # Sample random starting points (Gaussian noise) for flow trajectories
        field_start  = torch.randn_like(field_end)
        
        # Compute intermediate state at time r using linear interpolation
        # This is the standard flow matching interpolation: x_r = (1-r)*x_0 + r*x_1
        field_r      = self.flow(field_start, field_end, r, masked_batch)
        
        # Predict velocity field at current state and time
        pred = self.forward(field_r, y, r, masked_batch)
        
        # Compute flow matching loss
        self.loss(pred, field_start, field_end, masked_batch)
        return self.losses['train_loss']
    
    def loss(
        self,
        pred:         torch.Tensor, # Predicted velocity field (num_nodes, num_features)
        field_start:  torch.Tensor, # Starting state x_0 (num_nodes, num_features)
        field_end:    torch.Tensor, # Target state x_1 (num_nodes, num_features)
        masked_batch: torch.Tensor, # Batch indices (num_nodes,)
    ) -> torch.Tensor:
        """
        Compute flow matching loss.
        
        The target velocity field is analytically known in flow matching:
        w_r(x_r) = x_1 - x_0 (constant velocity field for linear interpolation)
        
        Loss encourages the learned velocity field to match this target.
        """
        # Compute analytical target velocity field (advection field)
        advection_field = self.advection_field(field_start, field_end)
        
        # Mean squared error between predicted and target velocity fields
        # Averaged over nodes within each graph, then over graphs in batch
        # self.losses['train_loss'] = scatter(
        #     (pred - advection_field).pow(2).mean(dim=-1), 
        #     masked_batch, 
        #     dim    = 0, 
        #     reduce = 'mean',
        # ).mean()
        self.losses['train_loss'] = (pred - advection_field).pow(2).mean()