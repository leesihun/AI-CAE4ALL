"""Base class shared by VAE, SAR and the flow-matching Sampler.

AI-CAE4ALL port: upstream ``Model`` is a ``pytorch_lightning.LightningModule``
whose ``fit`` builds a Lightning ``Trainer`` with a wandb logger, a
``ModelCheckpoint`` and a learning-rate early-stopping callback. The suite runs
no Lightning anywhere else, so this file reimplements exactly the parts the
models use, on plain PyTorch:

* ``save_hyperparameters()``  -> the constructor arguments of the most-derived
  class are captured into ``self.hparams`` (``_capture_hparams``);
* ``self.device``             -> first parameter's device;
* ``load_from_checkpoint``    -> rebuild from the stored hparams, load weights;
* ``Trainer.fit``             -> ``fit`` below: Adam, ReduceLROnPlateau on the
  epoch-mean ``train_loss`` (stepped once per epoch, ``eps=0``), gradient
  accumulation, optional gradient clipping, and upstream's
  ``LearningRateEarlyStopping`` (stop at the start of an epoch whose learning
  rate fell below ``lr_threshold``, checked every ``lr_check_every_n_epoch``).

The models' own ``forward`` / ``training_step`` / ``loss`` / ``sample`` code is
untouched; they still write their batch losses into ``self.losses``.
"""
import inspect
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from torch import nn

from .training_settings import TrainingSettings
from ..loader import DataLoader


def _encode_hparam(value):
    """Make a constructor argument storable with ``torch.load(weights_only=True)``."""
    if isinstance(value, type) and issubclass(value, nn.Module):
        return {'__nn_class__': value.__name__}
    if isinstance(value, tuple):
        return [_encode_hparam(v) for v in value]
    if isinstance(value, list):
        return [_encode_hparam(v) for v in value]
    return value


def _decode_hparam(value):
    if isinstance(value, dict) and set(value) == {'__nn_class__'}:
        return getattr(nn, value['__nn_class__'])
    if isinstance(value, list):
        return [_decode_hparam(v) for v in value]
    return value


class Model(nn.Module, ABC):
    """Abstract base class for models.

    Child classes implement ``forward`` and ``training_step``; ``training_step``
    must leave the batch losses in ``self.losses`` (``train_loss`` at least).
    """

    def __init__(self) -> None:
        super().__init__()
        self.hparams = self._capture_hparams()
        self.losses = {}

    # ------------------------------------------------------------------ hparams
    def _capture_hparams(self) -> dict:
        """Lightning's ``save_hyperparameters`` without Lightning.

        Every subclass calls ``super().__init__()`` as its first statement, so
        the outermost ``__init__`` frame whose ``self`` is this object still
        holds exactly the arguments it was called with.
        """
        frame = inspect.currentframe()
        target = None
        try:
            f = frame.f_back
            while f is not None:
                if f.f_code.co_name == '__init__' and f.f_locals.get('self') is self:
                    target = f
                f = f.f_back
            if target is None:
                return {}
            sig = inspect.signature(type(self).__init__)
            hp = {}
            for name, param in sig.parameters.items():
                if name == 'self' or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    continue
                if name in target.f_locals:
                    hp[name] = _encode_hparam(target.f_locals[name])
            return hp
        finally:
            del frame, target

    @classmethod
    def load_from_checkpoint(cls, path: str, map_location='cpu', **overrides):
        """Rebuild a model from a checkpoint written by ``save_checkpoint``.

        ``overrides`` replace stored constructor arguments (the SAR stage uses
        it to point ``vae_ckpt`` at the VAE file of *this* run).
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        hparams = {k: _decode_hparam(v) for k, v in ckpt['hyper_parameters'].items()}
        hparams.update(overrides)
        model = cls(**hparams)
        model.load_state_dict(ckpt['state_dict'])
        return model

    def checkpoint_dict(self, **extra) -> dict:
        out = {
            'model_class': type(self).__name__,
            'hyper_parameters': dict(self.hparams),
            'state_dict': self.state_dict(),
        }
        out.update(extra)
        return out

    # ------------------------------------------------------------------ helpers
    @property
    def device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        for b in self.buffers():
            return b.device
        return torch.device('cpu')

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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

        ``epoch_callback(epoch, stats, model)`` runs after every epoch (logging,
        validation, periodic checkpoint). ``resume`` is a
        ``general_modules.resume_state.TrainingResume``; ``start_state`` is what
        its ``load()`` returned (optimizer/scheduler/RNG/epoch);
        ``extra_state()`` returns caller-owned entries (the cumulative wall
        time) saved into the same resume state.
        Returns the final per-epoch stats.
        """
        ts = training_settings
        device = self.device
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
        for epoch in range(start_epoch, ts.max_epochs):
            lr = optimizer.param_groups[0]['lr']
            # Upstream LearningRateEarlyStopping (on_train_epoch_start).
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
                        torch.nn.utils.clip_grad_norm_(self.parameters(), ts.gradient_clip_val)
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
                    'epoch': epoch,
                    **(extra_state() if extra_state is not None else {}),
                })
        return stats

    # ------------------------------------------------------------------ abstract
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def training_step(self, *args, **kwargs):
        pass
