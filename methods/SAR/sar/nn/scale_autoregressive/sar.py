import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from copy import deepcopy
import time
from typing import Union
import warnings

from .masker import HierarchicalMasker
from ..models import VAE
from ..flow_matching import Sampler
from .condition_encoder import ConditionEncoder
from ..model import Model
from ..modules.transolver import Transolver
from ..modules.fnn import FNN
from ...graph import Graph
from ...loader import Collater


class SAR(Model):
    """
    Scale Autoregressive Model (SAR) for Physical Field Generation
    
    SAR generates physical fields autoregressively across a hierarchy of spatial resolutions,
    proceeding in a coarse-to-fine manner. At each step, finer-scale predictions are conditioned
    on predicted values at coarser scales, enabling fast and accurate generation by concentrating
    computational effort on coarser scales where uncertainty is greatest.
    
    The model implements the joint likelihood factorization:
    p(S_{1:K}) = ∏_{k=1}^K p(S_k | X, V_c, Γ, S_{1:k-1})
    
    where S_k are field values at scale k, X are spatial coordinates, V_c are conditioning
    attributes, and Γ indicates node scales.
    
    Architecture consists of three main components:
    1. Condition Encoder: Embeds geometric and physical information into node-wise features
    2. Autoregressive Module: Determines how coarse-scale predictions influence next scale
    3. Sampler: Probabilistic model for generating field values using flow matching
    """
    
    def __init__(
        self,
        num_iters:          int,                # Number of autoregressive steps (K scales)
        num_cond_layers:    int,                # Layers in condition encoder
        num_ar_layers:      int,                # Layers in autoregressive module  
        num_sampler_layers: int,                # Layers in sampler
        d_model:            int,                # Hidden dimension
        num_slices:         int,                # Number of slices for physics-informed attention
        vae_ckpt:           str = None,         # Path to pre-trained VAE for latent-space SAR
        in_features:        int = None,         # Input feature dimension (ignored if using VAE)
        dim:                int = 2,            # Spatial dimension
        cond_features:      int = 0,            # Conditioning feature dimension
        d_cond:             int = None,         # Condition encoder output dimension
        sampler_arch:       str = 'transolver', # Sampler architecture
        d_sampler:          int = None,         # Sampler hidden dimension
        d_r:                int = None,         # Time embedding dimension for flow matching
        num_heads:          int = 8,            # Attention heads
        dropout:            float = 0.0,        # Dropout rate
        noise_std:          float = 0.0,        # Gaussian noise std for training robustness
        sampler_batch_size: int = 1,            # Batch size for sampler training
    ) -> None:
        """
        Initialize Scale Autoregressive Model.
        
        Args:
            num_iters: Number of resolution scales (autoregressive steps)
            vae_ckpt: Path to pre-trained autoencoder for latent-space operation
            Other args: Network architecture parameters
        """
        super().__init__()

        # Store hyperparameters
        self.num_iters          = num_iters
        self.num_cond_layers    = num_cond_layers
        self.num_ar_layers      = num_ar_layers
        self.num_sampler_layers = num_sampler_layers
        self.d_model            = d_model
        self.num_slices         = num_slices
        self.vae_ckpt           = vae_ckpt
        self.in_features        = in_features
        self.dim                = dim
        self.cond_features      = cond_features
        self.d_cond             = d_model if d_cond is None else d_cond
        self.sampler_arch       = sampler_arch
        self.d_sampler          = d_model if d_sampler is None else d_sampler
        self.d_r                = d_model if d_r is None else d_r
        self.num_heads          = num_heads
        self.dropout            = dropout
        self.noise_std          = noise_std
        self.sampler_batch_size = sampler_batch_size
        self.is_latent          = True if self.vae_ckpt is not None else False
        self.cache = {}  # For caching conditioning vectors during multi-sample generation
        
        # Latent-Space SAR: Load and freeze pre-trained autoencoder
        # This reduces denoising steps at finest scale and removes residual noise
        if self.is_latent:
            self.vae = VAE.load_from_checkpoint(self.vae_ckpt)
            self.vae.eval()
            self.vae.to(self.device)
            # Freeze autoencoder parameters - it's used only for encoding/decoding
            for param in self.vae.parameters():
                param.requires_grad = False
            # Set features based on autoencoder's latent space
            if self.in_features is not None:
                warnings.warn("`in_features` is ignored when using an autoencoder", UserWarning)
            self.in_features = self.vae.latent_features
            self.ref_scale   = self.vae.num_scales  # Reference scale from VAE
        else:
            assert self.in_features is not None, "`in_features` must be provided when not using an autoencoder"
            self.ref_scale = 1
        self.out_features  = self.in_features

        # Component 1: Condition Encoder
        # Processes entire node set to embed geometric and physical information
        # Evaluated once per sample and can be cached for multiple samples
        self.cond_encoder = ConditionEncoder(
            dim           = self.dim,
            cond_features = self.cond_features,
            num_iters     = self.num_iters,
            out_features  = self.d_cond,
            ref_scale     = self.ref_scale,
            d_model       = self.d_model,
            num_layers    = self.num_cond_layers,
            num_heads     = self.num_heads,
            num_slices    = self.num_slices,
            dropout       = self.dropout,
        )

        # Hierarchical masker for managing autoregressive generation
        # Determines which nodes are known vs. unknown at each autoregressive step
        self.masker = HierarchicalMasker(self.num_iters, self.ref_scale)

        # Learnable embeddings for autoregressive steps and masking
        self.iter_embedding = nn.Parameter(torch.empty(self.num_iters, self.d_model))
        nn.init.xavier_normal_(self.iter_embedding)
        self.feature_encoder = nn.Linear(self.in_features, self.d_model)
        self.mask_embedding = nn.Parameter(torch.empty(1, self.d_model))
        nn.init.xavier_normal_(self.mask_embedding)

        # Component 2: Autoregressive Module  
        # Core component that determines how coarse-scale predictions influence next scale
        # Uses AdaLN-Zero conditioning on autoregressive step via iter_embedding
        self.ar_module = Transolver(
            in_features  = self.d_model + self.d_cond,
            out_features = self.d_model,
            num_layers   = self.num_ar_layers,
            d_model      = self.d_model,
            num_heads    = self.num_heads,
            num_slices   = self.num_slices,
            d_cond       = self.d_model,  # Conditioned on iter_embedding
            dropout      = self.dropout,
        )

        # Conditioning MLPs for flow matching
        # Transform spatial and conditioning information for the sampler
        self.cond_mlp_1 = FNN(
            in_features  = self.dim + self.num_iters,  # Position + scale one-hot
            layers_width = [self.d_model, self.d_model],
            activation   = nn.SELU,
            dropout      = self.dropout,
        )
        self.cond_mlp_2 = FNN(
            in_features  = self.d_cond,
            layers_width = [self.d_model, self.d_model],
            activation   = nn.SELU,
            dropout      = self.dropout,
        )

        # Component 3: Sampler
        # Probabilistic model using flow matching for generating field values
        # Evaluated multiple times per autoregressive step (once per denoising iteration)
        self.sampler = Sampler(
            num_features   = self.out_features,
            cond_features  = self.d_model * 2,  # Combined conditioning from MLPs above
            d_model        = self.d_sampler,
            d_r            = self.d_r,
            num_layers     = self.num_sampler_layers,
            num_heads      = self.num_heads,
            num_slices     = self.num_slices,
            dropout        = self.dropout,
            arch           = self.sampler_arch,
        )

    def forward(
        self,
        graph: Graph,
    ) -> torch.Tensor:
        """
        Forward pass for training. Implements one step of the autoregressive process.
        
        During training, we randomly select a scale k and compute loss L_k by:
        1. Encoding target to latent space (if using VAE)
        2. Computing conditioning vectors Y via condition encoder
        3. Sampling random iteration and creating masks for known/unknown nodes
        4. Running autoregressive module to get latent representations Z_k
        5. Preparing conditioning for flow matching sampler
        
        Returns: Conditioning vectors for the sampler (masked_y, masked_z, target, batch)
        """
        target = graph.target.clone()
        
        # Encode target features to latent space if using Latent-Space SAR
        if self.is_latent:
            self.vae.eval()
            target = self.vae.encode(graph, target)[0]
            # Reset edge index and batch after VAE encoding (may change due to multi-scale)
            graph.edge_index, graph.batch = getattr(graph, 'edge_index_1', graph.edge_index), getattr(graph, 'batch_1', graph.batch)

        # Create scale conditioning: one-hot encoding of scales beginning at reference scale
        scale = graph.scale[graph.scale >= self.ref_scale] - self.ref_scale + 1 # Scales begin at 1
        s = F.one_hot(scale - 1, num_classes=self.num_iters)
        pos = getattr(graph, f'pos_{self.ref_scale}', graph.pos)

        # Component 1: Condition Encoder
        # Compute global conditioning vectors Y = ConditionEncoder(X, V_c, Γ)
        # These capture both local features and global context across the domain
        y = self.cond_encoder(
            graph,
            pos,
            s,
        )

        # Sample random iteration numbers for each graph and get hierarchical masks
        # This implements the training strategy where we randomly select scale k
        iters, unknown_mask, target_mask = self.masker.sample(graph)

        # Crop to include only known and target nodes (exclude fully unknown nodes)
        crop_mask = torch.logical_or(~unknown_mask, target_mask)
        pos          = pos        [crop_mask]
        s            = s          [crop_mask]
        y            = y          [crop_mask]
        target       = target     [crop_mask]
        target_mask  = target_mask[crop_mask]
        batch        = graph.batch[crop_mask]

        # Add Gaussian noise for training robustness against error accumulation
        # This is crucial for autoregressive models to handle prediction errors
        noise = torch.randn_like(target) * self.noise_std
        u = self.feature_encoder(target + noise)

        # Mask target nodes with learnable mask embedding
        # This indicates to the autoregressive module that these values are unknown
        u[target_mask] = self.mask_embedding.repeat(target_mask.sum(), 1)

        # Component 2: Autoregressive Module  
        # Compute Z_k = AR(k, Y_{1:k}, S_{1:k-1}) - latent representations for target scale
        # The iter_embedding conditions the network on current autoregressive step k
        masked_z = self.ar_module(
            torch.cat([u, y], dim=1),  # Concatenate features and conditioning
            batch, 
            self.iter_embedding[iters],  # Step conditioning via AdaLN-Zero
        )[target_mask]  # Extract only target node representations

        # Prepare conditioning for Component 3: Sampler
        masked_batch = batch[target_mask]
        
        # Generate conditioning vectors for flow matching
        # Combines spatial information (pos, scale) and global context (y)
        masked_y = (
            self.cond_mlp_1(
                torch.cat([
                    pos[target_mask],  # Spatial coordinates
                    s[target_mask],    # Scale one-hot encoding
                ], dim=1)
            ) + 
            self.cond_mlp_2(
                y[target_mask]     # Global conditioning from condition encoder
            )
        )

        # Return conditioning vectors for sampler
        return masked_y, masked_z, target.clone()[target_mask], masked_batch
    
    def training_step(
        self,
        graph: Graph,
    ) -> torch.Tensor:
        """
        Training step implementing the flow-matching objective.
        
        Optimizes: L_k = E[||w_{k,r}(S_{k,r}) - u_{k,r}(S_{k,r} | X_k, Y_k, Z_k, Γ_k)||^2]
        where w_{k,r} is target flow field and u_{k,r} is neural approximation.
        """
        # Get conditioning vectors from forward pass
        masked_y, masked_z, masked_target, masked_batch = self(graph)
        
        # Component 3: Sampler training with flow matching objective
        # The sampler learns to transform Gaussian noise to target distribution
        self.sampler.training_step(
            masked_target,  # Target field values S_k
            torch.cat([     # Combined conditioning
                masked_z,   # From autoregressive module
                masked_y,   # From conditioning MLPs
            ], dim=1),
            masked_batch,
            self.sampler_batch_size,
        )
        self.loss()
        return self.losses['train_loss']
    
    def loss(self):
        """Aggregate loss from sampler (flow matching objective)."""
        self.losses['train_loss'] = self.sampler.losses['train_loss']

    @torch.no_grad()
    def sample(self, graph: Graph, steps: Union[int, list[int], dict] = 10,
               seed: int = None, verbose: bool = False) -> torch.Tensor:
        """
        Generate predictions using hierarchical autoregressive approach.
        
        Implements the full autoregressive generation process:
        S_{1:K} ~ ∏_{k=1}^K p(S_k | X, V_c, Γ, S_{1:k-1})
        
        Key insight: More denoising steps allocated to coarser scales (higher uncertainty),
        fewer steps to finer scales (benefit from strong conditioning).
        
        Args:
            graph: Input graph with node features and topology
            steps: Integration steps for flow matching per scale
                  - int: same steps for all scales  
                  - list: specific steps per scale
                  - dict: adaptive scheduling {'first': X, 'last': Y, 'schedule': 'cos'/'linear'}
            seed: Random seed for reproducibility
            verbose: Print progress information
            
        Returns:
            Predicted field values (in original space if using VAE)
        """
        if verbose:
            print(f"{self.num_iters}-scale autoregressive sampling in latent space...")
            global_timer_start = time.time()
        batch_size = graph.batch.max() + 1

        # Process steps parameter - adaptive allocation strategy
        if isinstance(steps, int):
            steps = [steps] * self.num_iters
        elif isinstance(steps, dict):
            # Adaptive scheduling: more steps for coarse scales, fewer for fine scales
            first    = steps['first']     # Steps for coarsest scale
            last     = steps['last']      # Steps for finest scale  
            schedule = steps.get('schedule', 'cos')
            if schedule == 'linear':
                steps = np.linspace(first, last, self.num_iters).round(0).astype(int)
            elif schedule == 'cos':
                # Cosine schedule concentrates more steps at coarse scales
                steps = np.array([first * np.cos(i * np.pi / 2 / (self.num_iters - 1)) + 
                                 last * (1 - np.cos(i * np.pi / 2 / (self.num_iters - 1))) 
                                 for i in range(self.num_iters)]).round(0).astype(int)
            else:
                raise ValueError(f"schedule must be one of 'linear' or 'cos'")
        elif isinstance(steps, list):
            pass
        else:
            raise ValueError(f"steps must be an int or a dict")
        assert all(step > 0 for step in steps), f"steps must be greater than 0"

        if seed is not None:
            torch.manual_seed(seed)
        self.eval()
        graph.to(self.device)
        batch_size = graph.batch.max() + 1

        # Prepare scale conditioning
        scale = graph.scale[graph.scale >= self.ref_scale] - self.ref_scale + 1 # Scales begin at 1
        s = F.one_hot(scale - 1, num_classes=self.num_iters)
        pos = getattr(graph, f'pos_{self.ref_scale}', graph.pos)

        # Use cached conditioning if available (for multi-sample generation)
        if self.cache.get('y', None) is not None:
            y = self.cache['y']
            graph.edge_index, graph.batch = getattr(graph, 'edge_index_1', graph.edge_index), getattr(graph, 'batch_1', graph.batch)
        else:
            # Component 1: Condition Encoder - compute global conditioning vectors
            y = self.cond_encoder(
                graph,
                pos,
                s,
            )

        # Initialize predictions in latent space
        pred = torch.zeros((pos.shape[0], self.out_features), device=self.device)

        # Hierarchical autoregressive generation: coarse-to-fine
        for i, iter_idx in enumerate(range(self.num_iters)):
            if verbose:
                print(f"Iteration: {iter_idx + 1}/{self.num_iters} ({steps[iter_idx]} steps)", end="\t")
                timer_start = time.time()

            iter = torch.full((batch_size,), iter_idx, device=self.device)

            # Get hierarchical masks: unknown_mask (not yet predicted), target_mask (to predict now)
            unknown_mask, target_mask = self.masker(graph, iter)

            # Crop to relevant nodes (known + target, exclude fully unknown)
            crop_mask = torch.logical_or(~unknown_mask, target_mask)
            pos_c         = pos        [crop_mask]
            s_c           = s          [crop_mask]
            y_c           = y          [crop_mask]
            target_mask_c = target_mask[crop_mask]
            batch_c       = graph.batch[crop_mask]

            # Encode current predictions (known values + zeros for targets)
            u = self.feature_encoder(pred[crop_mask])

            # Mask target nodes to indicate unknown status
            u[target_mask_c] = self.mask_embedding.repeat(target_mask_c.sum(), 1)

            # Component 2: Autoregressive Module
            # Extract latent representations Z_k for target nodes
            masked_z = self.ar_module(
                torch.cat([u, y_c], dim=1),
                batch_c, 
                self.iter_embedding[iter],  # Condition on autoregressive step
            )[target_mask_c]

            masked_batch = batch_c[target_mask_c]

            # Generate conditioning vectors for sampler (matching training exactly)
            masked_y = (
                self.cond_mlp_1(
                    torch.cat([
                        pos_c[target_mask_c],
                        s_c[target_mask_c],
                    ], dim=1)
                ) + 
                self.cond_mlp_2(
                    y_c[target_mask_c]
                )
            )

            # Component 3: Sampler - generate field values for current scale
            # Uses flow matching with adaptive number of denoising steps
            pred[target_mask] = self.sampler.sample(
                torch.cat([
                    masked_z,    # Autoregressive conditioning
                    masked_y,    # Spatial + global conditioning
                ], dim=1),
                masked_batch,
                steps[i],       # Adaptive step allocation
            )

            if verbose:
                timer_end = time.time()
                print(f"Time elapsed: {(timer_end - timer_start) * 1000:.2f} ms")

        # Decode from latent space to physical space if using Latent-Space SAR
        if self.is_latent:
            pred = self.vae.decode(graph, pred)

        if verbose:
            global_timer_end = time.time()
            print(f"Total time elapsed: {(global_timer_end - global_timer_start) * 1000:.2f} ms \n")
        return pred
    
    @torch.no_grad()    
    def sample_n(self, num_samples: int, graph: Graph, batch_size: int = 1,
                cache: bool = True, *args, **kwargs) -> torch.Tensor:
        """
        Generate multiple sample predictions from same input graph.
        
        Efficient multi-sample generation with conditioning vector caching.
        Since condition encoder Y = ConditionEncoder(X, V_c, Γ) depends only on 
        geometry and physical parameters (not stochastic), it can be cached when
        generating multiple samples for the same domain.
        
        Args:
            num_samples: Number of samples to generate
            graph: Input graph with node features and topology
            batch_size: Number of samples to generate in parallel (0 for sequential)
            cache: Whether to cache computation results for efficiency
            *args, **kwargs: Additional arguments for sample method
            
        Returns:
            Stacked sample predictions (num_nodes, num_samples, num_fields)
        """
        samples = []

        # Pre-compute and cache conditioning vectors for efficiency
        if cache:
            self.eval()
            graph.to(self.device)
            # Create conditioning encoding vectors
            scale = graph.scale[graph.scale >= self.ref_scale] - self.ref_scale + 1  # Scales begin at 1
            s = F.one_hot(scale - 1, num_classes=self.num_iters)
            pos = getattr(graph, f'pos_{self.ref_scale}', graph.pos)
            # Compute once and cache
            y = self.cond_encoder(
                graph,
                pos,
                s,
            )
            # Reset edge index and batch after potential VAE operations
            graph.edge_index, graph.batch = getattr(graph, 'edge_index_1', graph.edge_index), getattr(graph, 'batch_1', graph.batch)
            # Store in cache for reuse across samples
            self.cache['y'] = y

        # Generate samples (parallel or sequential)
        if batch_size > 1:
            # Parallel sampling: process multiple samples simultaneously
            collater = Collater()
            num_evals = num_samples // batch_size + (num_samples % batch_size > 0)
            produced = 0
            for _ in tqdm(range(num_evals), desc=f"Generating {num_samples} samples", leave=False, position=0):
                # AI-CAE4ALL port: upstream subtracted len(samples), which counts
                # BATCHES, so a num_samples that is not a multiple of batch_size
                # returned extra samples (10 at batch 4 -> 12). Count samples.
                current_batch_size = min(batch_size, num_samples - produced)
                produced += current_batch_size
                batch = collater.collate([deepcopy(graph) for _ in range(current_batch_size)])
                if cache:
                    self.cache['y'] = y.repeat(current_batch_size, 1)
                sample = self.sample(batch, *args, **kwargs)
                sample = torch.stack(sample.chunk(current_batch_size, dim=0), dim=1)
                samples.append(sample)
        else:
            # Sequential sampling: one sample at a time
            for _ in tqdm(range(num_samples), desc=f"Generating {num_samples} samples", leave=False, position=0):
                sample = self.sample(graph, *args, **kwargs)
                samples.append(sample.unsqueeze(1))

        # Clear cache to free memory
        self.cache = {}
        return torch.cat(samples, dim=1) 