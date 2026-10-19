import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import scatter, to_dense_batch

# AI-CAE4ALL port, two runtime switches (not hyperparameters, not saved):
#
# DENSE_ATTENTION  The reference PhysicsAttention below materialises the
#     outer product x (x) slice_weights as a [num_nodes, heads, slices, d_head]
#     tensor twice per layer (slicing and de-slicing). At d_model 128 / 8 heads
#     / 64 slices that is 8192 floats per node: ~2.9 GB per layer for one 87k-node
#     ex3 shell, before autograd keeps it. ``_dense_forward`` computes the same
#     two contractions as batched einsums over a padded [B, N_max, ...] layout,
#     which never forms that tensor. The arithmetic is identical up to float
#     summation order (tests/test_sar_port.py::
#     test_dense_attention_matches_reference); the reference path is kept and
#     used whenever the batch vector is not sorted.
# USE_CHECKPOINTING  torch.utils.checkpoint around each Transolver block while
#     training (recompute in backward instead of storing activations).
DENSE_ATTENTION = True
USE_CHECKPOINTING = False


def set_runtime_options(dense_attention=None, use_checkpointing=None):
    global DENSE_ATTENTION, USE_CHECKPOINTING
    if dense_attention is not None:
        DENSE_ATTENTION = bool(dense_attention)
    if use_checkpointing is not None:
        USE_CHECKPOINTING = bool(use_checkpointing)

from .fnn import FNN
from .adaln import AdaLNZero


class PhysicsAttention(nn.Module):
    def __init__(
        self,
        d_in:     int,
        d_out:    int,
        d_model:  int, 
        n_heads:  int,
        n_slices: int,
        dropout:  float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads."
        # Hyperparameters
        self.d_in     = d_in
        self.d_out    = d_out
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.n_slices = n_slices
        self.d_head   = d_model // n_heads
        self.scale    = self.d_head ** -0.5
        self.dropout  = nn.Dropout(dropout)  
        # Linear input layer
        self.to_input = nn.Linear(self.d_in, d_model)
        # Linear projection for adaptive temperature
        self.to_temperature = nn.Linear(self.d_head, 1)    
        # Linear projection for slice weights
        self.to_slice = nn.Linear(self.d_head, self.n_slices)
        self.softmax  = nn.Softmax(dim=-1)
        torch.nn.init.orthogonal_(self.to_slice.weight)
        # Linear layers for computing query, key, and value for attention.
        self.to_q = nn.Linear(self.d_head, self.d_head, bias=False)
        self.to_k = nn.Linear(self.d_head, self.d_head, bias=False)
        self.to_v = nn.Linear(self.d_head, self.d_head, bias=False)
        # Output projection.
        self.to_out = nn.Sequential(nn.Linear(self.d_model, self.d_out), nn.Dropout(dropout))

    def forward(
        self,
        x:     torch.Tensor,
        batch: torch.LongTensor,
    ) -> torch.Tensor:
        if DENSE_ATTENTION and (batch.numel() < 2 or bool((batch[1:] >= batch[:-1]).all())):
            return self._dense_forward(x, batch)
        return self._reference_forward(x, batch)

    def _dense_forward(
        self,
        x:     torch.Tensor,
        batch: torch.LongTensor,
    ) -> torch.Tensor:
        """Same maths as ``_reference_forward`` without the [N, H, S, D] tensor."""
        total_nodes = x.size(0)
        x = self.to_input(x).view(total_nodes, self.n_heads, self.d_head)
        temp = self.to_temperature(x).exp()
        slice_weights = self.softmax(self.to_slice(x) / temp)              # [N, H, S]
        # Padded rows carry zero weights, so they add nothing to any sum.
        x_d, mask = to_dense_batch(x, batch)                               # [B, M, H, D]
        w_d, _ = to_dense_batch(slice_weights, batch)                      # [B, M, H, S]
        slice_norm = w_d.sum(dim=1)                                        # [B, H, S]
        slice_token = torch.einsum('bmhs,bmhd->bhsd', w_d, x_d)            # [B, H, S, D]
        slice_token = slice_token / (slice_norm.unsqueeze(-1) + 1e-5)
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)                # [B, H, S, D]
        out_d = torch.einsum('bmhs,bhsd->bmhd', w_d, out_slice_token)      # [B, M, H, D]
        out_x = out_d[mask]                                                # [N, H, D]
        return self.to_out(out_x.reshape(total_nodes, self.d_model))

    def _reference_forward(
        self,
        x:     torch.Tensor,
        batch: torch.LongTensor,
    ) -> torch.Tensor:
        total_nodes = x.size(0)
        # Input layer and create heads
        x = self.to_input(x).view(total_nodes, self.n_heads, self.d_head) # shape: [B*N, n_heads, d_head]
        # Adaptive temperature
        temp = self.to_temperature(x).exp()  # shape: [B*N, n_heads, 1]
        # Compute slice weights
        slice_weights = self.softmax(self.to_slice(x) / temp) # shape: [B*N, n_heads, slice_num]
        # Compute normalization factor per graph
        slice_norm = scatter(slice_weights, batch, dim=0) # shape: [B, n_heads, n_slices]
        # Aggregate node features into slice tokens per graph
        slice_token = scatter(
            x.unsqueeze(-2) * slice_weights.unsqueeze(-1),
            batch,
        )  # shape: [B, n_heads, n_slices, d_head]
        slice_token = slice_token / (slice_norm.unsqueeze(-1) + 1e-5)
        # Self-attention among slice tokens.
        q_slice_token = self.to_q(slice_token) # shape: [B, n_heads, n_slices, d_head]
        k_slice_token = self.to_k(slice_token) # shape: [B, n_heads, n_slices, d_head]
        v_slice_token = self.to_v(slice_token) 
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token) # shape: [B, n_heads, n_slices, d_heads]
        # Deslice
        out_slice_token_expanded = out_slice_token[batch.squeeze()] # shape: [B*N, n_heads, n_slices, d_head]
        # Weighted sum over slice tokens.
        out_x = torch.sum(out_slice_token_expanded * slice_weights.unsqueeze(-1), dim=-2) # shape: [B*N, n_heads, d_head]
        return self.to_out(out_x.view(total_nodes, self.d_model)) # shape: [B*N, d_out]


class TransolverBlock(nn.Module):
    def __init__(
        self,
        d_in:     int,
        d_out:    int,
        d_model:  int,
        n_heads:  int,
        n_slices: int,
        d_cond:   int   = 0,
        dropout:  float = 0.0,
    ) -> None:
        super().__init__()
        # Hyperparameters
        self.d_in     = d_in
        self.d_out    = d_out
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.n_slices = n_slices
        self.d_cond   = d_cond
        self.dropout  = dropout
        # AdaLN
        if d_cond > 0:
            self.ln1 = AdaLNZero(d_model, d_cond)
            self.ln2 = AdaLNZero(d_model, d_cond)
        else:
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
        # Physics attention
        self.attention = PhysicsAttention(
            d_in     = d_in,
            d_out    = d_model,
            d_model  = d_model,
            n_heads  = n_heads,
            n_slices = n_slices,
            dropout  = dropout,
        )
        # Feedforward
        self.fnn = FNN(self.d_model, [self.d_model, self.d_model], activation=nn.GELU, dropout=self.dropout)

    def forward(
        self,
        x:     torch.Tensor,        # (num_nodes, d_model)
        batch: torch.LongTensor,    # (num_nodes)
        c:     torch.Tensor = None, # (batch_size, d_cond)
    ) -> torch.Tensor:
        # Multi-head attention
        if self.d_cond > 0:
            assert c is not None, "c must be provided if d_cond > 0."
            ln1_x, alpha = self.ln1(x, c, batch)
            x = x + alpha * self.attention(ln1_x, batch)
            # Feedforward
            ln2_x, alpha = self.ln2(x, c, batch)
            return x + alpha * self.fnn(ln2_x) # (num_nodes, d_model)
        else:
            ln1_x = self.ln1(x)
            x = x + self.attention(ln1_x, batch)
            # Feedforward  
            ln2_x = self.ln2(x)
            return x + self.fnn(ln2_x) # (num_nodes, d_model)
    

class Transolver(nn.Module):
    def __init__(
        self,
        in_features:  int,
        out_features: int,
        num_layers:   int,
        d_model:      int,
        num_heads:    int,
        num_slices:   int,
        d_cond:       int   = 0,
        dropout:      float = 0.0
    ) -> None:
        super().__init__()
        # Hyperparameters
        self.in_features  = in_features
        self.out_features = out_features
        self.num_layers   = num_layers
        self.d_model      = d_model
        self.num_heads    = num_heads
        self.num_slices   = num_slices
        self.d_cond       = d_cond
        self.dropout      = dropout
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.SELU(),
        )
        # Transformer encoder blocks
        self.layers = nn.ModuleList([
            TransolverBlock(
                d_in     = self.d_model,
                d_out    = self.d_model,
                d_model  = self.d_model,
                n_heads  = self.num_heads,
                n_slices = self.num_slices,
                d_cond   = self.d_cond,
                dropout  = self.dropout,
            ) for _ in range(self.num_layers)
        ])
        # Output layer
        self.output_layer = nn.Linear(d_model, out_features)

    def forward(
        self,
        x:     torch.Tensor,        # (num_nodes, in_features)
        batch: torch.LongTensor,    # (num_nodes,)
        c:     torch.Tensor = None, # (batch_size, d_cond)
    ) -> torch.Tensor:
        x = self.input_layer(x)
        for layer in self.layers:
            if USE_CHECKPOINTING and self.training and torch.is_grad_enabled():
                x = checkpoint(layer, x, batch, c, use_reentrant=False)
            else:
                x = layer(x, batch, c)
        return self.output_layer(x) # (num_nodes, out_features)
        
