import torch
import torch.nn as nn

from .adaln import AdaLNZero
from .fnn import FNN


class ResNetBlock(nn.Module):
    def __init__(
        self,
        d_model:    int,
        num_layers: int = 2,
        activation: nn.Module = nn.GELU,
        d_cond:     int   = 0,
        dropout:    float = 0.0,
    ) -> None:
        super().__init__()
        # Hyperparameters
        self.d_model    = d_model
        self.num_layers = num_layers
        self.activation = activation
        self.d_cond     = d_cond
        self.dropout    = dropout
        # AdaLN
        if d_cond > 0:
            self.ln = AdaLNZero(d_model, d_cond)
        else:
            self.ln = nn.LayerNorm(d_model)
        # Feedforward
        self.fnn = FNN(self.d_model, [self.d_model,] * self.num_layers, activation=self.activation, dropout=self.dropout)

    def forward(
        self,
        x:     torch.Tensor,        # (num_nodes, d_model)
        batch: torch.LongTensor,    # (num_nodes)
        c:     torch.Tensor = None, # (batch_size, d_cond)
    ) -> torch.Tensor:
        # Multi-head attention
        if self.d_cond > 0:
            assert c is not None, "c must be provided if d_cond > 0."
            ln_x, alpha = self.ln(x, c, batch)
            return x + alpha * self.fnn(ln_x) # (num_nodes, d_model)
        else:
            ln_x = self.ln(x)
            return x + self.fnn(ln_x) # (num_nodes, d_model)
    

class ResNet(nn.Module):
    def __init__(
        self,
        in_features:  int,
        out_features: int,
        num_layers:   int,
        d_model:      int,
        activation:   nn.Module = nn.GELU,
        d_cond:       int   = 0,
        dropout:      float = 0.0
    ) -> None:
        super().__init__()
        # Hyperparameters
        self.in_features  = in_features
        self.out_features = out_features
        self.num_layers   = num_layers
        self.d_model      = d_model
        self.activation   = activation
        self.d_cond       = d_cond
        self.dropout      = dropout
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.SELU(),
        )
        # Transformer encoder blocks
        self.layers = nn.ModuleList([
            ResNetBlock(
                d_model    = self.d_model,
                num_layers = self.num_layers,
                activation = self.activation,
                d_cond     = self.d_cond,
                dropout    = self.dropout,
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
            x = layer(x, batch, c)
        return self.output_layer(x) # (num_nodes, out_features)
        
