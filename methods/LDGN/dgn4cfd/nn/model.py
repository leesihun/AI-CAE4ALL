"""Base class shared by the VGAE and the latent diffusion model.

AI-CAE4ALL port: upstream ``Model.fit`` built its own optimiser loop around a
``TrainingSettings`` object with a TensorBoard writer, a tqdm bar, a periodic
``.chk`` file and optional mixed precision. The suite owns logging, resume and
checkpoints in ``training_profiles``, so ``fit`` here is the loop every other
port uses (methods/SAR/sar/nn/model.py), with upstream's optimisation recipe
unchanged:

* Adam over the trainable parameters at ``lr``;
* gradient-norm clipping (upstream ``grad_clip={'epoch': 0, 'limit': 1}``
  clips from the first epoch on, which is ``gradient_clip_val 1``);
* ``ReduceLROnPlateau(factor, patience, eps=0)`` stepped once per epoch on the
  epoch-mean training loss (upstream ``scheduler['loss'] = 'training'``);
* stop at the start of an epoch whose learning rate fell below the threshold
  (upstream ``stopping``).

Each model's ``training_step(graph)`` holds what upstream's loop did per batch
and leaves its batch losses in ``self.losses``. Checkpoints are plain
``weights_only``-loadable dicts: the activation class in ``arch`` is stored by
name and a ``DiffusionProcess`` by its constructor arguments.
"""
import time
from typing import Callable, Optional

import torch
from torch import nn

from .training_settings import TrainingSettings
from ..graph import Graph
from ..loader import DataLoader


def encode_arch(value):
    """Make an ``arch`` dict storable with ``torch.load(weights_only=True)``."""
    if isinstance(value, type) and issubclass(value, nn.Module):
        return {'__nn_class__': value.__name__}
    if isinstance(value, dict):
        return {k: encode_arch(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_arch(v) for v in value]
    return value


def decode_arch(value):
    if isinstance(value, dict):
        if set(value) == {'__nn_class__'}:
            return getattr(nn, value['__nn_class__'])
        return {k: decode_arch(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_arch(v) for v in value]
    return value


def diffusion_process_params(dp) -> dict:
    return {
        'num_steps': int(dp.num_steps),
        'schedule_type': str(dp.schedule_type),
        'beta_start': float(dp.beta_start),
        'beta_end': float(dp.beta_end),
        'max_beta': float(dp.max_beta),
    }


class Model(nn.Module):
    r"""Base class for all the GNN models.

    Args:
        arch (dict, optional): Dictionary with the model architecture.
        weights (str, optional): Path of a state-dict file.
        checkpoint (str or dict, optional): A checkpoint written from
            ``checkpoint_dict`` (path or the loaded dict).
        device (torch.device, optional): Device where the model is loaded.
    """

    def __init__(
        self,
        arch:       dict         = None,
        weights:    str          = None,
        checkpoint               = None,
        device:     torch.device = torch.device('cpu'),
    ) -> None:
        super().__init__()
        self._init_device = torch.device(device)
        self.losses = {}
        self.load_model(arch, weights, checkpoint)

    # ------------------------------------------------------------------ device
    @property
    def device(self) -> torch.device:
        # Upstream stored ``self.device`` and kept it in sync by hand; here it
        # follows the parameters, so ``model.to(...)`` is enough.
        for p in self.parameters():
            if not nn.parameter.is_lazy(p):
                return p.device
        for b in self.buffers():
            if not nn.parameter.is_lazy(b):
                return b.device
        return self._init_device

    @device.setter
    def device(self, value) -> None:
        self._init_device = torch.device(value)

    # To be overwritten
    @property
    def num_fields(self) -> int:
        """Returns the number of output fields."""
        pass

    # ------------------------------------------------------------------ loading
    def load_model(self, arch, weights, checkpoint):
        """Builds the model from ``arch`` (+ optional ``weights``) or from a checkpoint."""
        if arch is not None and checkpoint is None:
            self.load_arch(arch)
            self.to(self._init_device)
            if weights is not None:
                self.load_state_dict(torch.load(weights, map_location=self._init_device, weights_only=True))
        elif arch is None and weights is None and checkpoint is not None:
            if not isinstance(checkpoint, dict):
                checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if 'diffusion_process' in checkpoint:
                from .diffusion.diffusion_process import DiffusionProcess
                self.diffusion_process = DiffusionProcess(**checkpoint['diffusion_process'])
            if 'learnable_variance' in checkpoint:
                self.learnable_variance = checkpoint['learnable_variance']
            self.load_arch(decode_arch(checkpoint['arch']))
            self.load_state_dict(checkpoint['weights'])
            self.to(self._init_device)
        return

    def checkpoint_dict(self, **extra) -> dict:
        """What upstream ``save_checkpoint`` stored (``arch``, ``weights``, the
        diffusion process, ``learnable_variance``) minus the optimiser state,
        which the suite's resume file owns; plus caller-owned ``extra``."""
        out = {
            'model_class': type(self).__name__,
            'arch': encode_arch(self.arch),
            'weights': self.state_dict(),
        }
        if hasattr(self, 'diffusion_process'):
            out['diffusion_process'] = diffusion_process_params(self.diffusion_process)
        if hasattr(self, 'learnable_variance'):
            out['learnable_variance'] = self.learnable_variance
        out.update(extra)
        return out

    # To be overwritten
    def load_arch(self, arch: dict):
        """Defines the hyper-parameters of the model."""
        pass

    # To be overwritten
    def forward(self, graph: Graph):
        pass

    # To be overwritten
    def training_step(self, graph: Graph) -> torch.Tensor:
        """One batch of upstream's training loop: returns the scalar loss to
        back-propagate and leaves the batch losses in ``self.losses``."""
        raise NotImplementedError

    def fit_state(self) -> dict:
        """Mutable training state outside the weights (resume)."""
        return {}

    def load_fit_state(self, state: dict) -> None:
        pass

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        training_settings: TrainingSettings,
        train_dataset,
        epoch_callback: Optional[Callable] = None,
        resume=None,
        start_state: Optional[dict] = None,
        extra_state: Optional[Callable] = None,
    ) -> dict:
        """Train with upstream's optimisation recipe.

        ``epoch_callback(epoch, stats, model)`` runs after every epoch.
        ``resume`` is a ``general_modules.resume_state.TrainingResume``;
        ``start_state`` is what its ``load()`` returned; ``extra_state()``
        returns caller-owned entries saved into the same resume state.
        Returns the final per-epoch stats.
        """
        ts = training_settings
        device = self.device
        self.prepare_fit(ts)
        optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=ts.lr)
        scheduler = None
        if ts.lr_scheduler_factor is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=ts.lr_scheduler_factor,
                patience=ts.lr_scheduler_patience, eps=0)

        loader = DataLoader(
            dataset=train_dataset,
            batch_size=ts.batch_size,
            shuffle=True,
            num_workers=ts.num_workers,
            pin_memory=device.type == 'cuda',
            persistent_workers=ts.num_workers > 0,
            generator=ts.generator,
        )

        start_epoch = 0
        if start_state is not None:
            self.load_state_dict(start_state['model'])
            optimizer.load_state_dict(start_state['optimizer'])
            if scheduler is not None and start_state.get('scheduler') is not None:
                scheduler.load_state_dict(start_state['scheduler'])
            self.load_fit_state(start_state.get('fit_state') or {})
            if ts.generator is not None and start_state.get('loader_generator') is not None:
                # Persistent workers draw their base seed when the iterator is
                # first built: build it before restoring any RNG stream.
                if ts.num_workers > 0:
                    iter(loader)
                ts.generator.set_state(start_state['loader_generator'])
            elif ts.num_workers > 0:
                iter(loader)
            start_epoch = int(start_state['epoch']) + 1
            if resume is not None:
                resume.restore_rng()

        stats = {}
        accum = max(1, int(ts.accumulate_grad_batches))
        num_batches = len(loader)
        params = [p for p in self.parameters() if p.requires_grad]
        for epoch in range(start_epoch, ts.max_epochs):
            lr = optimizer.param_groups[0]['lr']
            if ts.lr_threshold and epoch % ts.lr_check_every_n_epoch == 0 and lr < ts.lr_threshold:
                print(f"Learning rate {lr:.3e} below threshold {ts.lr_threshold:.3e}: stopping at epoch {epoch}")
                stats['stopped_early'] = True
                break
            t0 = time.time()
            self.train()
            sums, count = {}, 0
            optimizer.zero_grad(set_to_none=True)
            for b, graph in enumerate(loader):
                graph = graph.to(device)
                loss = self.training_step(graph)
                (loss / accum).backward()
                if (b + 1) % accum == 0 or (b + 1) == num_batches:
                    if ts.gradient_clip_val:
                        nn.utils.clip_grad_norm_(params, ts.gradient_clip_val)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for name, value in self.losses.items():
                    v = float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
                    sums[name] = sums.get(name, 0.0) + v
                count += 1
            stats = {name: total / max(count, 1) for name, total in sums.items()}
            stats['lr'] = lr
            stats['epoch'] = epoch
            stats['elapsed'] = time.time() - t0
            if scheduler is not None:
                scheduler.step(stats.get('train_loss', 0.0))
            if epoch_callback is not None:
                epoch_callback(epoch, stats, self)
            if resume is not None and resume.enabled and resume.due(epoch, ts.max_epochs):
                resume.save({
                    'model': self.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler is not None else None,
                    'loader_generator': ts.generator.get_state() if ts.generator is not None else None,
                    'fit_state': self.fit_state(),
                    'epoch': epoch,
                    **(extra_state() if extra_state is not None else {}),
                })
        return stats

    def prepare_fit(self, training_settings: TrainingSettings) -> None:
        """Hook run once before the optimiser is built (step sampler, loss)."""
        pass

    # ------------------------------------------------------------------ helpers
    @property
    def num_params(self):
        """Returns the number of parameters."""
        return sum(p.numel() for p in self.parameters() if not nn.parameter.is_lazy(p))

    @property
    def num_learnable_params(self):
        """Returns the number of learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad and not nn.parameter.is_lazy(p))
