from dataclasses import dataclass, fields
from typing import Any, Optional

import torch


@dataclass
class TrainingSettings:
    """Training settings consumed by ``Model.fit``.

    AI-CAE4ALL port: upstream's dataclass also carried the Lightning/wandb
    plumbing (``ckpt_dir``, ``logs_dir``, ``logs_project``, ``accelerator``,
    ``devices``, ``strategy``, ``precision``, ``save_every_n_epochs``,
    ``ckpt_path``, ``log_every_n_steps``, ``wandb_run_id``). Checkpoints, logs
    and resume are owned by ``training_profiles`` here, so only the fields that
    shape the optimisation remain -- with upstream's names and defaults.
    ``lr_scheduler_monitor`` is gone because the only monitored quantity
    upstream ever used is ``train_loss``. ``generator`` seeds the loader's
    shuffling when ``training_seed`` is set.
    """
    model_name: str
    lr: float

    max_epochs: int = int(1e6)
    batch_size: int = 32
    num_workers: int = 0

    gradient_clip_val: Optional[float] = None
    accumulate_grad_batches: int = 1

    lr_scheduler_factor: Optional[float] = None
    lr_scheduler_patience: Optional[int] = None

    lr_threshold: float = 0
    lr_check_every_n_epoch: int = 1

    generator: Optional[torch.Generator] = None

    def __post_init__(self):
        if self.lr_threshold is not None:
            assert self.lr_threshold <= self.lr, \
                "Learning rate threshold must be less than or equal to the initial learning rate"

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
