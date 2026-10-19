import math
import torch
from torch import nn


class SinusoidalPositionEmbedding(nn.Module):
    r"""Defines a sinusoidal embedding like in the paper "Attention is All You Need" (https://arxiv.org/abs/1706.03762).

    Args:
        dim (int): The dimension of the embedding.
        theta (float, optional): The theta parameter of the sinusoidal embedding. Defaults to 10000.
    """

    def __init__(
        self,
        dim: int,
        theta: float = 10000.,
        ) -> None:
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even."
        self.dim = dim
        self.theta = theta

    def forward(
        self,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the embedding of position `r`."""    
        device = r.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb) # Dimensions: [dim/2]
        emb = r.unsqueeze(-1) * emb.unsqueeze(0) # Dimensions: [batch_size, dim/2]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) # Dimensions: [batch_size, dim]
        return emb
    

class SinusoidalTimeEmbedding(nn.Module):
    r"""Defines a sinusoidal embedding for continuos time in [t_min, t_max].

    Args:
        dim (int): The dimension of the embedding.
        t_min (float): The minimum time.
        t_max (float): The maximum time.
    """

    def __init__(
        self,
        dim:   int,
        t_min: float = 0.0,
        t_max: float = 1.0,
    ) -> None:
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even."
        self.dim = dim
        self.t_min = t_min
        self.t_max = t_max

    def forward(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the embedding of time `t`."""  
        assert t.dim() == 1, "Time must be one-dimensional."
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000.) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        t   = (t - self.t_min) / (self.t_max - self.t_min) * 1000.
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb