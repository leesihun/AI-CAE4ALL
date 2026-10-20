from dataclasses import dataclass, fields
from typing import Any, Optional

import torch


@dataclass
class TrainingSettings:
    """Training settings consumed by ``Model.fit``.

    AI-CAE4ALL port: upstream's class also carried the run plumbing
    (``folder``, ``checkpoint``, ``tensor_board``, ``chk_interval``,
    ``mixed_precision``, ``device``) and took the loss as a callable. Logs,
    checkpoints and resume belong to ``training_profiles`` here, so what is
    left shapes the optimisation, under the names the SAR port uses:

    * ``epochs``                -> ``max_epochs``
    * ``grad_clip['limit']``    -> ``gradient_clip_val`` (from epoch 1 on)
    * ``scheduler['factor']``   -> ``lr_scheduler_factor``
    * ``scheduler['patience']`` -> ``lr_scheduler_patience``
    * ``stopping``              -> ``lr_threshold``
    * ``step_sampler``          -> ``step_sampler`` ('importance' | 'uniform')
    * ``VaeLoss(kl_reg)``       -> ``kl_reg``
    * ``HybridLoss(lambda_vlb)``-> ``lambda_vlb``

    ``scheduler['loss']`` is gone because upstream's diffusion loop accepts
    only the training loss. ``generator`` seeds the loader's shuffling when
    ``training_seed`` is set.
    """
    model_name: str
    lr: float

    max_epochs: int = 1
    batch_size: int = 1
    num_workers: int = 0

    gradient_clip_val: Optional[float] = None
    accumulate_grad_batches: int = 1

    lr_scheduler_factor: Optional[float] = None
    lr_scheduler_patience: Optional[int] = None

    lr_threshold: float = 0
    lr_check_every_n_epoch: int = 1

    kl_reg: float = 0.0
    lambda_vlb: float = 0.001
    step_sampler: str = 'importance'

    generator: Optional[torch.Generator] = None

    def __post_init__(self):
        if self.lr_threshold is not None:
            assert self.lr_threshold <= self.lr, \
                "Learning rate threshold must be less than or equal to the initial learning rate"
        assert self.step_sampler in ('importance', 'uniform'), \
            f"step_sampler must be 'importance' or 'uniform', got {self.step_sampler!r}"

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def print(self) -> None:
        print('=' * 72)
        print('Training Settings'.center(72))
        print('=' * 72)
        for f in fields(self):
            if f.name == 'generator':
                continue
            print(f"  {f.name:<24} {getattr(self, f.name)}")
        print('=' * 72)
