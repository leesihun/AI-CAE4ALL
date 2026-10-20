# AI-CAE4ALL port: only the losses an LDGN uses are kept (VaeLoss for the
# autoencoder, HybridLoss for the latent diffusion model, and their helpers).
# MseLoss, GaussianNLLLoss, BayesianLoss, GmmLoss and FlowMatchingLoss belong to
# baselines that are not vendored; dropping FlowMatchingLoss also drops the
# import of the (not vendored) flow_matching package.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.utils import scatter

from .diffusion.diffusion_model import DiffusionModel
from .model import Model
from ..graph import Graph


class VaeLoss(nn.Module):
    def __init__(self, kl_reg: float = 0.0):
        super().__init__()
        self.kl_reg = kl_reg # KL regularization coefficient

    def forward(
            self,
            model: Model,
            graph: Graph,
    ) -> torch.Tensor:
        v, _, mean, logvar = model(graph)
        loss = {}
        loss['reconstruction_loss'] = F.mse_loss(v, graph.target)
        if self.kl_reg > 0.0:
            kl_divergence = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
            loss['kl_divergence'] = self.kl_reg * kl_divergence
        # Add all the terms
        return sum(loss.values())


class HybridLoss(nn.Module):
    """Hybrid loss function for diffusion models from the paper Improved Denoising Diffusion Probabilistic Models (https://arxiv.org/abs/2102.09672)."""

    def __init__(
        self,
        lambda_vlb: float = 0.001,
    ) -> None:
        super().__init__()
        self.lambda_vlb = lambda_vlb

    def forward(
        self,
        model: DiffusionModel,
        graph: Graph,
    ) -> torch.Tensor:
        assert model.learnable_variance, 'HybridLoss requires learnable_variance = `True`'
        true_posterior_mean, true_posterior_variance = model.diffusion_process.get_posterior_mean_and_variance(graph.field_start, graph.field_r, graph.batch, graph.r)
        # Compute the model output and the MSE loss between eps and noise
        model_noise, model_v = model(graph)
        se = (model_noise - graph.noise)**2 # Dimension: (num_nodes, num_output_features)
        mse_term = batch_wise_mean(se, graph.batch) # Dimension: (batch_size)
        # Freeze eps and compute the VLB loss
        frozen_output = (model_noise.detach(), model_v)
        model_posterior_mean, model_posterior_variance = model.get_posterior_mean_and_variance_from_output(frozen_output, graph)
        vlb_term = self.lambda_vlb * self.vlb_loss(graph, true_posterior_mean, true_posterior_variance, model_posterior_mean, model_posterior_variance) # Dimension: (batch_size)
        # AI-CAE4ALL: keep the two terms so the epoch log can plot them apart
        model.loss_terms = {'mse_loss': mse_term.detach(), 'vlb_loss': vlb_term.detach()}
        return mse_term + vlb_term # Dimension: (batch_size)
    
    @staticmethod
    def vlb_loss(
        graph: Graph,
        true_posterior_mean:      torch.Tensor,
        true_posterior_variance:  torch.Tensor,
        model_posterior_mean:     torch.Tensor,
        model_posterior_variance: torch.Tensor
    ) -> torch.Tensor:
        """Compute the KL divergence between the true and the model posterior"""
        kl = normal_kl_divergence(true_posterior_mean, true_posterior_variance, model_posterior_mean, model_posterior_variance) # Dimension: (num_nodes, num_features)
        kl = batch_wise_mean(kl, graph.batch)/math.log(2.0) # Dimension (batch_size)
        if (graph.r == 0).any():
            decoder_nll = F.gaussian_nll_loss(model_posterior_mean, graph.field_start, model_posterior_variance, reduction='none') # Dimension: (num_nodes, num_features)
            decoder_nll = batch_wise_mean(decoder_nll, graph.batch) # Dimension (batch_size)
            kl = torch.where((graph.r == 0), decoder_nll, kl)
        return kl


def normal_kl_divergence(
    mean1:     torch.Tensor,
    variance1: torch.Tensor,
    mean2:     torch.Tensor,
    variance2: torch.Tensor
) -> torch.Tensor:
    """Compute the KL divergence between two normal distributions."""
    return 0.5 * (torch.log(variance2) - torch.log(variance1) + variance1 / variance2 + (mean1 - mean2)**2 / variance2 - 1)


def batch_wise_mean(
    field: torch.Tensor,
    batch: torch.LongTensor,
) -> torch.Tensor:
    r"""Compute the batch-wise mean of a field.
    
        Args:
            field (Tensor): The field. Dimension: (num_nodes, num_features).
            batch (LongTensor): The batch vector. Dimension: (num_nodes).

        Returns:
            Tensor: The batch-wise mean. Dimension: (batch_size).
    """
    assert field.dim() == 1 or field.dim() == 2, 'field must be one- or two-dimensional'
    if field.dim() == 2:
        field = field.mean(dim=1) # Dimension: (num_nodes)
    batch_size = batch.max().item() + 1
    return scatter(field, batch, dim=0, dim_size=batch_size, reduce='mean') # Dimension: (batch_size)
