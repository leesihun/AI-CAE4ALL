import torch
from typing import Tuple, Union
from abc import abstractmethod
from copy import deepcopy

from .diffusion_process import DiffusionProcess, DiffusionProcessSubSet
from .step_sampler import ImportanceStepSampler, UniformStepSampler
from ..model import Model
from ...graph import Graph
from ...loader import Collater


# AI-CAE4ALL port changes in this file:
# * upstream ``fit`` (its own optimiser loop, TensorBoard, tqdm, ``.chk``
#   files) is replaced by ``Model.fit`` plus ``training_step`` below, which is
#   upstream's per-batch body verbatim (step sampling, diffusion, loss,
#   importance-sampler update, weighted mean);
# * the step sampler's loss history is part of the resume state;
# * ``sample_n``: the batched path counted generated *batches* against
#   ``num_samples`` (``num_samples - len(samples)``), so the last batch was
#   never shortened and more samples than asked could come back; the
#   one-by-one path reused one graph that ``sample`` mutates (its
#   ``edge_index``/``batch`` become the coarsest level's), so the second call
#   ran on a corrupted graph. Both fixed; tqdm removed.


class DiffusionModel(Model):
    r"""Abstract class for diffusion models. The forward pass must be implemented in the derived class.

    Args:
        diffusion_process (DiffusionProcess, optional): The diffusion process. It must be provided if we are not loading a checkpoint. Defaults to `None`.
        learnable_variance (bool, optional): If `True`, the variance is learnable. If we are loading a checkpoint, this value must be the same as the one in the checkpoint.
            If `False`, the model output the noise, as in "Denoising Diffusion Probabilistic Models". If `True`, the model output the noise and the variance, as in "Improved Denoising Diffusion Probabilistic Models".
            Defaults to `False`.
    """

    def __init__(
        self,
        diffusion_process:  DiffusionProcess = None,
        learnable_variance: bool             = None,
        *args,
        **kwargs
    ) -> None:
        self.learnable_variance = learnable_variance
        super().__init__(*args, **kwargs)
        # This may overwrite the diffusion_process and learnable_variance if we are loading a checkpoint.
        # So, we need to check if the provided values are the same as the ones in the model.
        if diffusion_process is None and not hasattr(self, 'diffusion_process'):
            raise RuntimeError('diffusion_process must be provided')
        elif diffusion_process is not None:
            if hasattr(self, 'diffusion_process'):
                print('Warning: diffusion_process is provided, but it is already defined in the model.')
                print('The provided diffusion_process will be used.')
            self.diffusion_process = diffusion_process
        if learnable_variance is None and not hasattr(self, 'learnable_variance'):
            raise RuntimeError('learnable_variance must be provided')
        if learnable_variance is not None and learnable_variance != self.learnable_variance:
            raise RuntimeError('learnable_variance is different from the one in the provided checkpoint')
        self.step_sampler = None
        self.criterion = None

    @property
    def is_latent_diffusion(self):
        return hasattr(self, 'autoencoder')

    # ------------------------------------------------------------------ training
    def prepare_fit(self, training_settings) -> None:
        from ..losses import HybridLoss
        sampler = ImportanceStepSampler if training_settings.step_sampler == 'importance' else UniformStepSampler
        self.step_sampler = sampler(num_diffusion_steps=self.diffusion_process.num_steps)
        self.criterion = HybridLoss(lambda_vlb=training_settings.lambda_vlb)

    def fit_state(self) -> dict:
        s = self.step_sampler
        if isinstance(s, ImportanceStepSampler):
            return {'loss_history': torch.from_numpy(s._loss_history.copy()),
                    'loss_counts': torch.from_numpy(s._loss_counts.copy())}
        return {}

    def load_fit_state(self, state: dict) -> None:
        s = self.step_sampler
        if isinstance(s, ImportanceStepSampler) and 'loss_history' in state:
            s._loss_history = state['loss_history'].numpy().copy()
            s._loss_counts = state['loss_counts'].numpy().copy()

    def training_step(self, graph: Graph) -> torch.Tensor:
        """Upstream ``fit``'s per-batch body.

        In eval mode (the validation pass) the step is drawn uniformly from
        torch's RNG with unit weights and the importance sampler is neither
        read nor updated, so a validation pass changes no training state."""
        if self.criterion is None:
            from ..losses import HybridLoss
            self.criterion = HybridLoss()
        device = graph.pos.device
        batch_size = graph.batch.max().item() + 1
        if self.is_latent_diffusion:
            graph = self.autoencoder.transform(graph)
        # Sample a batch of random diffusion steps
        if self.training and self.step_sampler is not None:
            graph.r, sample_weight = self.step_sampler(batch_size, device)  # (batch_size), (batch_size)
        else:
            graph.r = torch.randint(0, self.diffusion_process.num_steps, (batch_size,), device=device)
            sample_weight = torch.ones(batch_size, device=device)
        # Diffuse the solution/target field
        graph.field_start = graph.x_latent_target if self.is_latent_diffusion else graph.target
        graph.field_r, graph.noise = self.diffusion_process(
            field_start    = graph.field_start,
            r              = graph.r,
            batch          = graph.batch,
            dirichlet_mask = None if self.is_latent_diffusion else getattr(graph, 'dirichlet_mask', None)
        )  # Shapes (num_nodes, num_fields), (num_nodes, num_fields)
        # Compute the loss for each sample in the batch
        self.loss_terms = {}
        loss = self.criterion(self, graph)  # Dimension: (batch_size)
        # Update the loss-aware diffusion-step sampler
        if self.training and isinstance(self.step_sampler, ImportanceStepSampler):
            self.step_sampler.update(graph.r.cpu().numpy(), loss.detach().cpu().numpy())
        # Compute the weighted loss over the batch
        loss = (loss * sample_weight).mean()
        self.losses = {'train_loss': loss.detach()}
        # Same weighting as the loss, so the terms add up to train_loss
        for name, term in self.loss_terms.items():
            self.losses[name] = (term * sample_weight).mean()
        return loss

    @abstractmethod
    def forward(self, graph: Graph) -> torch.Tensor:
        """Forward pass of the model. This method must be implemented for each model."""
        pass

    def get_posterior_mean_and_variance_from_output(
        self,
        model_output:      Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        graph:             Graph,
        diffusion_process: DiffusionProcess = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the posterior mean and variance from the model output.

        Args:
            model_output (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]): The model output.
                This can be the noise ("Denoising Diffusion Probabilistic Models") or a tuple with the noise and v ("Improved Denoising Diffusion Probabilistic Models")
            graph (Graph): The graph.
            diffusion_process (DiffusionProcess, optional): The diffusion process. If `None`, the diffusion process of the model is used. Defaults to `None`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The posterior mean and variance.
        """
        dp = self.diffusion_process if diffusion_process is None else diffusion_process
        if isinstance(model_output, tuple):
            assert self.learnable_variance, 'The model output is a tuple, but learnable_variance=False'
        else:
            assert not self.learnable_variance, 'The model output is not a tuple, but learnable_variance=True'
        # Pre-compute needed coefficients
        betas_r = dp.get_index_from_list(dp.betas, graph.batch, graph.r)
        sqrt_one_minus_alphas_cumprod_r = dp.get_index_from_list(dp.sqrt_one_minus_alphas_cumprod, graph.batch, graph.r)
        sqrt_recip_alphas_r = dp.get_index_from_list(dp.sqrt_recip_alphas, graph.batch, graph.r)
        if self.learnable_variance:
            eps, v = model_output
            v = (v + 1) / 2
            # For min_log we use the clipped posterior variance to avoid nan values in the backward pass
            min_log = dp.get_index_from_list(dp.posterior_log_variance_clipped, graph.batch, graph.r)
            max_log = torch.log(dp.get_index_from_list(dp.betas, graph.batch, graph.r))
            log_variance = v * max_log + (1 - v) * min_log
            variance = torch.exp(log_variance)
        else:
            eps = model_output
            variance = dp.get_index_from_list(dp.posterior_variance, graph.batch, graph.r)
        mean = sqrt_recip_alphas_r * (graph.field_r - betas_r * eps / sqrt_one_minus_alphas_cumprod_r)
        return mean, variance

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample(
        self,
        graph:            Graph,
        steps:            list[int]    = None,
        dirichlet_values: torch.Tensor = None,
    ) -> torch.Tensor:
        """Sample from the model.

        Args:
            graph (Graph): The graph. It is modified in place (its ``edge_index``
                and ``batch`` become the coarsest level's for a latent model).
            steps (list[int], optional): The steps to sample. If `None`, all steps are sampled. Defaults to `None`.
            dirichlet_values (torch.Tensor, optional): The Dirichlet boundary conditions. If `None`, no Dirichlet boundary conditions are applied. Defaults to `None`.

        Returns:
            torch.Tensor: The sample.
        """
        if steps is not None:
            assert all([isinstance(s, int) for s in steps]), 'steps must be a list of integers'
            # Sort the steps in ascending order
            steps = sorted(steps)
            # We cannot have repeated steps and the last step must be smaller than the number of steps in the "full" diffusion process
            assert len(steps) == len(set(steps)), 'steps must not have repeated values'
            assert steps[-1] < self.diffusion_process.num_steps, 'The last step in steps must be smaller than the number of steps in the "full" diffusion process'
        self.eval()
        device = self.device
        if not hasattr(graph, 'batch') or graph.batch is None:
            graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
        if graph.pos.device != device:
            graph.to(device)
        if dirichlet_values is not None:
            dirichlet_values = dirichlet_values.to(device)
        batch_size = graph.batch.max().item() + 1
        # Get the latent features if the model is a latent diffusion model
        if self.is_latent_diffusion:
            c_latent_list, e_latent_list, edge_index_list, batch_list = self.autoencoder.cond_encoder(
                graph,
                torch.cat([f for f in [graph.get('loc'), graph.get('glob'), graph.get('omega')] if f is not None], dim=1),
                torch.cat([f for f in [graph.get('edge_attr'), graph.get('edge_cond')] if f is not None], dim=1),
                graph.edge_index
            )
            graph.c_latent   = c_latent_list  [-1].clone()
            graph.e_latent   = e_latent_list  [-1].clone()
            graph.edge_index = edge_index_list[-1].clone()
            graph.batch      = batch_list     [-1].clone()
        # Use the spaced diffusion process if it is defined
        dp = DiffusionProcessSubSet(self.diffusion_process, steps) if steps is not None else self.diffusion_process
        # Sample the noisy solution
        graph.field_r = torch.randn(graph.batch.size(0), self.num_fields, device=device)  # Shape (num_nodes, num_fields)
        # Take into account the dirichlet boundary conditions if they are defined and if we are not using a latent diffusion model
        if hasattr(graph, 'dirichlet_mask') and not self.is_latent_diffusion:
            assert dirichlet_values is not None, 'dirichlet_values must be provided if graph has dirichlet_mask'
            assert dirichlet_values.shape == (graph.num_nodes, self.num_fields), f'dirichlet_values.shape must be (num_nodes, num_fields), but it is {dirichlet_values.shape}'
            dirichlet_field_r, _ = dp(
                field_start    = dirichlet_values,
                r              = torch.tensor(batch_size * [dp.num_steps - 1], dtype=torch.long, device=device),
                batch          = graph.batch,
                dirichlet_mask = graph.dirichlet_mask
            )
            graph.field_r = graph.field_r * (~graph.dirichlet_mask) + dirichlet_field_r * graph.dirichlet_mask
        # Denoise 'field_r' step-by-step
        for r in dp.steps[::-1]:
            graph.r = torch.full((batch_size,), r, dtype=torch.long, device=device)
            # Compute the posterior mean and variance
            model_output = self(graph)
            mean, variance = self.get_posterior_mean_and_variance_from_output(model_output, graph, dp)
            if hasattr(graph, 'dirichlet_mask') and not self.is_latent_diffusion:
                gaussian_noise = torch.randn_like(mean) * (~graph.dirichlet_mask)
            else:
                gaussian_noise = torch.randn_like(mean)
            graph.field_r = mean + torch.sqrt(variance) * gaussian_noise
        # Decode the denoised latent features
        if self.is_latent_diffusion:
            return self.autoencoder.decode(
                graph           = graph,
                v_latent        = graph.field_r,
                c_latent_list   = c_latent_list,
                e_latent_list   = e_latent_list,
                edge_index_list = edge_index_list,
                batch_list      = batch_list,
                dirichlet_mask  = graph.dirichlet_mask if hasattr(graph, 'dirichlet_mask') else None,
                v_0             = dirichlet_values     if hasattr(graph, 'dirichlet_mask') else None,
            )
        else:
            return graph.field_r

    @torch.no_grad()
    def sample_n(
        self,
        num_samples:      int,
        graph:            Graph,
        dirichlet_values: torch.Tensor = None,
        batch_size:       int = 0,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Sample `num_samples` samples from the model.

        Args:
            num_samples (int): The number of samples.
            graph (Graph): The graph (not modified).
            dirichlet_values (torch.Tensor, optional): The Dirichlet boundary conditions. If `None`, no Dirichlet boundary conditions are applied. Defaults to `None`.
            batch_size (int, optional): Number of samples to generate in parallel. If `batch_size < 2`, the samples are generated one by one. Defaults to `0`.

        Returns:
            torch.Tensor: The samples. Dimension: (num_nodes, num_samples, num_fields)
        """
        samples, done = [], 0
        if batch_size > 1:
            collater = Collater()
            while done < num_samples:
                current_batch_size = min(batch_size, num_samples - done)
                batch = collater.collate([deepcopy(graph) for _ in range(current_batch_size)])
                sample = self.sample(batch, *args, dirichlet_values=dirichlet_values.repeat(current_batch_size, 1) if dirichlet_values is not None else None, **kwargs)
                # Split base on the batch index
                sample = torch.stack(sample.chunk(current_batch_size, dim=0), dim=1)
                samples.append(sample)
                done += current_batch_size
            return torch.cat(samples, dim=1)
        else:
            for _ in range(num_samples):
                samples.append(self.sample(deepcopy(graph), *args, dirichlet_values=dirichlet_values, **kwargs))
            return torch.stack(samples, dim=1)  # Dimension: (num_nodes, num_samples, num_fields)
