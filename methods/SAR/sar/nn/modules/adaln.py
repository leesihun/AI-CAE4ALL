import torch
import torch.nn as nn

from .fnn import FNN


class AdaLNZero(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_emb:   int,
        eps:     float = 1e-5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_emb   = d_emb
        self.eps     = eps
        self.gamma = nn.Parameter(torch.ones ([1, d_model]))
        self.beta  = nn.Parameter(torch.zeros([1, d_model]))
        self.alpha = nn.Parameter(torch.zeros([1, d_model]))
        self.adaptive_gamma_beta_alpha = FNN(
            in_features  = self.d_emb,
            layers_width = (self.d_model, 3 * d_model),
            activation   = nn.SiLU,
        )
        # Set caching to False
        self.caching(False)

    def caching(
        self,
        enable: bool,
    ) -> None:
        self.caching_enabled, self.cached_gamma, self.cached_beta, self.cached_alpha = enable, None, None, None

    def forward(
        self, 
        x:     torch.Tensor,     # (num_nodes,  d_model)
        emb:   torch.Tensor,     # (batch_size, d_emb)
        batch: torch.LongTensor, # (num_nodes,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() == 2, "The input tensor must have two dimensions."
        assert emb.dim() == 2, "The embedding tensor must have two dimensions."
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var (dim=-1, unbiased=False, keepdim=True)
        x = (x - mean) / (var + self.eps).sqrt()
        # Caching
        if self.caching_enabled:
            if self.cached_gamma is None:
                adaptive_gamma, adaptive_beta, adaptive_alpha = self.adaptive_gamma_beta_alpha(emb).chunk(3, dim=-1)
                gamma = self.gamma + adaptive_gamma
                beta  = self.beta  + adaptive_beta
                alpha = self.alpha + adaptive_alpha
                self.cached_gamma, self.cached_beta, self.cached_alpha = gamma, beta, alpha
            else:
                gamma, beta, alpha = self.cached_gamma, self.cached_beta, self.cached_alpha
        else:
            adaptive_gamma, adaptive_beta, adaptive_alpha = self.adaptive_gamma_beta_alpha(emb).chunk(3, dim=-1)
            gamma = self.gamma + adaptive_gamma # (batch_size, d_model)
            beta  = self.beta  + adaptive_beta  # (batch_size, d_model)
            alpha = self.alpha + adaptive_alpha # (batch_size, d_model)
        return x * gamma[batch] + beta[batch], alpha[batch] # (num_nodes, d_model), (num_nodes, d_model)