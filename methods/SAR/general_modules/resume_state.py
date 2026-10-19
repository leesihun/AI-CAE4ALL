"""Mid-training resume, opted into with `resume_training true`.

A training that is cut off (the campaign runner stopped, the machine went
down) otherwise starts again from epoch 0. With `resume_training true` the
trainer writes a resume state beside its checkpoint, `<checkpoint>.resume`,
at the first epoch boundary after every `resume_interval_minutes` (default
15), and on start continues from a state that this same config wrote. The
final checkpoint is still written only by a run that reaches its last epoch,
so a cut-off run never looks finished; once it is on disk the resume state
is removed.

A state holds everything the next epoch reads -- weights, EMA, optimizer,
scheduler, AMP scaler, the epoch counter, the RNG streams and the trainer's
own trackers -- so the resumed run continues the same schedule rather than
restarting it. A state written under a different config is refused, not
silently reused. A stage trained on top of another one's checkpoint (an LC or
flow prior over a frozen VAE) names that file as `upstream`: once it has been
rewritten the state is moved to `<checkpoint>.resume.stale` and the stage
starts over. Single-process runs only: the multi-GPU entry points reject the
key (`reject_multi_process`).
"""

import hashlib
import os
import random
import time

import numpy as np
import torch

# Keys that legitimately differ between two launches of the same run.
_VOLATILE_KEYS = frozenset({'gpu_ids', 'log_dir', 'resume_training', 'resume_interval_minutes'})
DEFAULT_INTERVAL_MINUTES = 15.0


def config_fingerprint(config) -> str:
    """sha256 over the config's keys and values, minus launch-specific ones
    (`gpu_ids`, the resume keys, and runtime keys starting with `_`)."""
    items = sorted((str(k), repr(v)) for k, v in config.items()
                   if k not in _VOLATILE_KEYS and not str(k).startswith('_'))
    return hashlib.sha256(repr(items).encode('utf-8')).hexdigest()


def reject_multi_process(config, where: str) -> None:
    if bool(config.get('resume_training', False)):
        raise ValueError(f'resume_training is implemented for single-GPU training only, not {where}; '
                         'drop the key or run on one GPU')


def _rng_state() -> dict:
    state = {'python': random.getstate(), 'numpy': np.random.get_state(),
             'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state()
    return state


def _set_rng_state(state: dict) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['cuda'])


def _file_identity(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return [st.st_size, st.st_mtime_ns]


class TrainingResume:
    """The resume state of one training stage, kept at `<checkpoint_path>.resume`.

    load()   -> the saved dict (None when off or nothing to resume); call it
                after the model, optimizer and loaders are built, and call
                restore_rng() right before the first resumed epoch
    due()    -> True at most once per interval, never on the last epoch
    save()   -> atomic write of the caller's dict plus RNG and fingerprint
    finish() -> removes the state once the final checkpoint is written
    """

    def __init__(self, config, checkpoint_path, upstream=()):
        self.enabled = bool(config.get('resume_training', False))
        self.path = f'{checkpoint_path}.resume' if checkpoint_path else None
        if self.enabled and not self.path:
            raise ValueError('resume_training needs a checkpoint path to keep its state beside')
        minutes = config.get('resume_interval_minutes', DEFAULT_INTERVAL_MINUTES)
        self.interval = max(0.0, float(minutes)) * 60.0
        self.fingerprint = config_fingerprint(config)
        self.upstream = [_file_identity(p) for p in upstream]
        self._last = time.monotonic()
        self._rng = None

    def load(self):
        if not self.enabled or not os.path.exists(self.path):
            return None
        state = torch.load(self.path, map_location='cpu', weights_only=False)
        if state.get('fingerprint') != self.fingerprint:
            raise RuntimeError(
                f'{self.path} was written under a different config; move it away to train '
                'from epoch 0, or restore the config it was written with')
        if state.get('upstream', []) != self.upstream:
            os.replace(self.path, self.path + '.stale')
            print(f'{self.path} was written on top of a checkpoint that has since been rewritten; '
                  f'moved it to {self.path}.stale and training this stage from epoch 0')
            return None
        self._rng = state.get('rng')
        print(f"Resuming from {self.path}: epoch {state['epoch']} was the last one completed "
              f"(saved {state.get('saved', '?')}); training continues at epoch {state['epoch'] + 1}")
        return state

    def restore_rng(self) -> None:
        if self._rng is not None:
            _set_rng_state(self._rng)
            self._rng = None

    def due(self, epoch: int, total_epochs: int) -> bool:
        if not self.enabled or epoch >= total_epochs - 1:
            return False
        return time.monotonic() - self._last >= self.interval

    def save(self, state: dict) -> None:
        state = dict(state, fingerprint=self.fingerprint, upstream=self.upstream, rng=_rng_state(),
                     saved=time.strftime('%Y-%m-%d %H:%M:%S'))
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = self.path + '.tmp'
        torch.save(state, tmp)
        os.replace(tmp, self.path)
        self._last = time.monotonic()

    def finish(self) -> None:
        if not self.enabled:
            return
        # A kill inside torch.save leaves a partial .tmp; the finished stage has
        # no use for either file.
        for path in (self.path + '.tmp', self.path):
            if os.path.exists(path):
                os.remove(path)


def resume_log_line(epoch: int, tag=None) -> str:
    """The line a resumed run appends to its epoch log. score_rank.py drops
    the logged epochs from `epoch` on (they are about to be redone) and keeps
    the rest, so the log still reads as one run."""
    return f"==== Resume{f' [{tag}]' if tag else ''} at epoch {epoch} {time.strftime('%Y-%m-%d %H:%M:%S')} ===="
