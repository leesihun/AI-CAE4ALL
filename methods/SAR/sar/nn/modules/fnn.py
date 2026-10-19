import torch
import torch.nn as nn
from typing import Callable, Tuple


class FNN(nn.Module):
    """Feedforward Neural Network with dropout after the activation function.

    Args:
        in_features (int): Number of input features.
        layers_width (Tuple[int]): Width of the hidden layers.
        activation (Callable): Activation function. Default: nn.ReLU.
        dropout (float): Dropout probability. Default: 0.0.
    """

    def __init__(
        self,
        in_features:  int,
        layers_width: Tuple[int], 
        activation:   Callable = nn.ReLU,
        dropout:      float = 0.0
    ) -> None:
        super().__init__()
        self.num_layers = len(layers_width)
        self.layers = nn.Sequential()
        # Input layer
        self.layers.add_module("linear_0",     nn.Linear(in_features, layers_width[0]))
        self.layers.add_module("activation_0", activation())
        # Hidden and output layers
        for l in range(1, self.num_layers):
            self.layers.add_module("linear_"     + str(l), nn.Linear(layers_width[l-1], layers_width[l]))
            self.layers.add_module("activation_" + str(l), activation())
        # Dropout layer
        if dropout > 0:
            self.layers.add_module("dropout", nn.Dropout(dropout))

    def reset_parameters(self):
        for item in self.layers:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.layers(x)
