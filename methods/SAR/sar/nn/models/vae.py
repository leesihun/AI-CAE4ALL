import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

from ..model import Model
from ...graph import Graph
from ..modules.mp import InteractionNetwork, DownMP, UpMP


class DownBlock(nn.Module):
    def __init__(
        self,
        scale:      int,
        depth:      int,
        fnns_depth: int,
        fnns_width: int,   
        activation: nn.Module,
        dropout:    float,
        aggr:       str = 'sum',
        dim:        int = 2,
    ) -> None:
        # Validate inputs
        assert scale >= 0, "The scale must be non-negative"
        assert depth > 0, "The depth must be positive"
        super().__init__()
        self.scale          = scale
        self.depth          = depth
        self.fnns_depth     = fnns_depth
        self.fnns_width     = fnns_width
        self.activation     = activation
        self.dropout        = dropout
        self.aggr           = aggr
        self.dim            = dim
        # Edge encoder
        self.edge_encoder = nn.Linear(self.dim, self.fnns_width)
        # MP blocks
        self.mp_blocks = nn.ModuleList([
            InteractionNetwork(
                in_node_features  = self.fnns_width,
                in_edge_features  = self.fnns_width,
                out_node_features = self.fnns_width,
                out_edge_features = self.fnns_width,
                fnns_depth        = self.fnns_depth,
                fnns_width        = self.fnns_width,
                activation        = self.activation,
                dropout           = self.dropout,
                aggr              = self.aggr,
            ) for _ in range(depth)
        ])
        # Pooling
        self.pooling = DownMP(
            scale            = scale, 
            dim              = self.dim,
            in_node_features = self.fnns_width,
            fnn_depth        = self.fnns_depth,
            fnn_width        = self.fnns_width,
            activation       = self.activation,
            dropout          = self.dropout,
            aggr             = self.aggr,
            encode_edges     = False,
            scalar_rel_pos   = True,
        )
    
    def forward(
        self,
        graph: Graph,
        v:     torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Apply the edge encoder
        e = getattr(graph, f'edge_attr_{self.scale + 1}') if self.scale > 0 else graph.edge_attr
        e = self.edge_encoder(e)
        # Apply the MP blocks
        for mp in self.mp_blocks:
            v, e = mp(v, e, graph.edge_index)
        # Apply the DownMP layer
        graph, v, _ = self.pooling(graph, v) # Also updates `graph.edge_index` and `graph.batch`
        return v


class UpBlock(nn.Module):
    def __init__(
        self,
        scale:      int,
        depth:      int,
        fnns_depth: int,
        fnns_width: int,         
        activation: nn.Module,
        dropout:    float,
        aggr:       str = 'sum',
        dim:        int = 2,
    ):
        # Validate inputs
        assert scale >= 0, "The scale must be non-negative"
        assert depth > 0,  "The depth must be positive"
        self.scale          = scale
        self.depth          = depth
        self.fnns_depth     = fnns_depth
        self.fnns_width     = fnns_width
        self.activation     = activation
        self.dropout        = dropout
        self.aggr           = aggr
        self.dim            = dim
        super().__init__()
        # Unpooling
        self.unpooling = UpMP(
            scale           = self.scale,
            dim             = self.dim,
            in_features     = self.fnns_width,
            fnn_depth       = self.fnns_depth,
            fnn_width       = self.fnns_width,
            activation      = self.activation,
            dropout         = self.dropout,
            skip_connection = False,
            scalar_rel_pos  = True,
        )
        # Edge encoder
        self.edge_encoder = nn.Linear(self.dim, self.fnns_width)
        # MP blocks
        self.mp_blocks = nn.ModuleList([
            InteractionNetwork(
                in_node_features  = fnns_width,
                in_edge_features  = fnns_width,
                out_node_features = fnns_width,
                out_edge_features = fnns_width,
                fnns_depth        = fnns_depth,
                fnns_width        = fnns_width,
                activation        = activation,
                dropout           = dropout,
                aggr              = aggr,
            ) for _ in range(depth)
        ])

    def forward(
        self,    
        graph: Graph,
        v:     torch.Tensor,
    ) -> torch.Tensor:
        # Unpooling
        graph, v = self.unpooling(
            graph,
            v,
            edge_index_skip = getattr(graph, f'edge_index_{self.scale + 1}'),
            batch_skip      = getattr(graph, f'batch_{self.scale + 1}'),
        ) # Also updates `graph.edge_index` and `graph.batch`
        # Apply the edge encoder
        e = getattr(graph, f'edge_attr_{self.scale + 1}') if self.scale > 0 else graph.edge_attr
        e = self.edge_encoder(e)
        # Apply the residual blocks
        for mp in self.mp_blocks:
            v, e = mp(v, e, graph.edge_index)
        return v


class VAE(Model):
    def __init__(
        self,
        in_features:     int,
        num_layers:      list[int],
        d_model:         int,
        dim:             Optional[int] = 2,
        latent_features: Optional[int] = None,
        activation:      Optional[nn.Module] = nn.SELU,
        dropout:         Optional[float] = 0.0,
        aggr:            Optional[str] = 'sum',
        kl_weight:       Optional[float] = 0.0,
        noise_std:       Optional[float] = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.num_layers = num_layers
        self.num_scales = len(self.num_layers)
        self.d_model = d_model
        self.dim = dim
        self.latent_features = in_features if latent_features is None else latent_features
        self.activation = activation
        self.dropout = dropout
        self.aggr = aggr
        self.kl_weight = kl_weight
        self.noise_std = noise_std
        # Encoder
        self.enc_node_encoder = nn.Linear(self.in_features, self.d_model)
        self.enc_down_blocks = nn.ModuleList([
            DownBlock(
                scale          = l,
                depth          = n,
                fnns_depth     = 2,
                fnns_width     = self.d_model,
                activation     = self.activation,
                dropout        = self.dropout,
                aggr           = self.aggr,
                dim            = self.dim,
            ) for l, n in enumerate(self.num_layers[:-1])
        ])
        self.enc_bottleneck_edge_encoder = nn.Linear(self.dim, self.d_model)
        self.enc_bottleneck_blocks = nn.ModuleList([
            InteractionNetwork(
                in_node_features  = self.d_model,
                in_edge_features  = self.d_model,
                out_node_features = self.d_model,
                out_edge_features = self.d_model,
                fnns_depth        = 2,
                fnns_width        = self.d_model,
                activation        = self.activation,
                dropout           = self.dropout,
                aggr              = self.aggr,
            ) for _ in range(self.num_layers[-1])
        ])
        self.enc_node_decoder = nn.Linear(self.d_model, self.latent_features * 2)
        self.enc_batch_norm = nn.BatchNorm1d(1, affine=False, momentum=0.001)
        self.enc_bn = lambda x: self.enc_batch_norm(x.reshape(-1, 1)).reshape(x.shape)
        # Decoder
        self.dec_node_encoder = nn.Linear(self.latent_features, self.d_model)
        self.dec_bottleneck_edge_encoder = nn.Linear(self.dim, self.d_model)
        self.dec_bottleneck_blocks = nn.ModuleList([
            InteractionNetwork(
                in_node_features  = self.d_model,
                in_edge_features  = self.d_model,
                out_node_features = self.d_model,
                out_edge_features = self.d_model,
                fnns_depth        = 2,
                fnns_width        = self.d_model,
                activation        = self.activation,
                dropout           = self.dropout,
                aggr              = self.aggr,
            ) for _ in range(self.num_layers[-1])
        ])
        self.dec_up_blocks = nn.ModuleList([
            UpBlock(
                scale          = self.num_scales - l - 2,
                depth          = n,
                fnns_depth     = 2,
                fnns_width     = self.d_model,
                activation     = self.activation,
                dropout        = self.dropout,
                aggr           = self.aggr,
                dim            = self.dim,
            ) for l, n in enumerate(self.num_layers[:-1][::-1])
        ])
        self.dec_node_decoder = nn.Linear(self.d_model, self.out_features)

    def encode(
        self,
        graph: Graph,
        v:     torch.Tensor,
    ) -> torch.Tensor:
        # Down blocks
        v = self.enc_node_encoder(v)
        for down_block in self.enc_down_blocks:
            v = down_block(graph, v)
        # Bottleneck blocks
        e = self.enc_bottleneck_edge_encoder(getattr(graph, f'edge_attr_{self.num_scales}', graph.edge_attr))
        for bottleneck_block in self.enc_bottleneck_blocks:
            v, _ = bottleneck_block(v, e, graph.edge_index)
        # Decode the latent features
        mean, logvar = self.enc_node_decoder(v).chunk(2, dim=-1)
        v_latent = mean + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
        # Normalise the latent features
        v_latent = self.enc_bn(v_latent)
        return v_latent, mean, logvar

    def decode(
        self,
        graph: Graph,
        v:     torch.Tensor,
    ) -> torch.Tensor:
        # Encode the node and edge features
        v = self.dec_node_encoder(v)
        e = self.dec_bottleneck_edge_encoder(getattr(graph, f'edge_attr_{self.num_scales}', graph.edge_attr))
        # Bottleneck blocks
        for bottleneck_block in self.dec_bottleneck_blocks:
            v, e = bottleneck_block(v, e, graph.edge_index)
        # Up blocks
        for up_block in self.dec_up_blocks:
            v = up_block(graph, v)
        # Decode the node features
        v = self.dec_node_decoder(v)
        return v 

    def forward(
        self,
        graph: Graph,
    ) -> torch.Tensor:
        # Encode the target
        v_latent, mean, logvar = self.encode(graph, graph.target.clone())
        # Add noise to the latent features
        if self.noise_std > 0:
            v_latent += torch.randn_like(v_latent) * self.noise_std
        # Decode the target
        v = self.decode(graph, v_latent)
        return v, v_latent, mean, logvar
    
    def training_step(
        self,
        graph: Graph,
    ) -> torch.Tensor:
        v, _, mean, logvar = self(graph)
        self.loss(v, graph.target, mean, logvar)
        return self.losses['train_loss']
    
    def loss(
        self,
        v: torch.Tensor,
        target: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        self.losses['reconstruction_loss'] = F.mse_loss(v, target)
        self.losses['kl_divergence_loss']  = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp()) if self.kl_weight > 0.0 else 0.0
        self.losses['train_loss']          = self.losses['reconstruction_loss'] + self.kl_weight * self.losses['kl_divergence_loss']