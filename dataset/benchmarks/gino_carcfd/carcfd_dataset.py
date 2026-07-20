"""Lazy HDF5 dataset for the isolated GINO CarCFD paper benchmark."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


def _decode_ids(dataset) -> list[str]:
    values = np.asarray(dataset)
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


class CarCFDPaperDataset(Dataset):
    """One graph per official manifest entry, without random re-splitting."""

    def __init__(self, path: str | Path, split: str):
        self.path = Path(path).resolve()
        self.split = str(split).lower()
        if self.split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}.")
        self._handle: h5py.File | None = None
        with h5py.File(self.path, "r") as handle:
            self.case_ids = _decode_ids(handle[f"splits/{self.split}_ids"])
            self.pressure_mean = float(handle.attrs["pressure_mean"])
            self.pressure_std = float(handle.attrs["pressure_std"])
            self.pressure_eps = float(handle.attrs.get("pressure_eps", 1.0e-7))
            self.resolution = int(handle.attrs["grid_resolution"])
            self.protocol = str(handle.attrs["benchmark_protocol"])
            self.diagnostic_only = bool(handle.attrs["diagnostic_only"])
            self.paper_target = float(handle.attrs["paper_target_mean_relative_l2"])
            expected = int(handle.attrs[f"converted_{self.split}_count"])
        if len(self.case_ids) != expected:
            raise ValueError(
                f"{self.split} manifest has {len(self.case_ids)} IDs but HDF5 declares {expected}."
            )

        # Minimal dataset metadata used to construct the immutable DataSpec.
        self.input_dim = 1
        self.output_dim = 1
        self.num_pos_features = 0
        self.use_node_types = False
        self.num_node_types = None
        self.operator_dim = 3
        self.active_axes = (0, 1, 2)
        self.has_sdf = False  # no per-mesh SDF; latent_sdf is benchmark-specific
        self.num_timesteps = 1

    def _h5(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self._handle.close()
            self._handle = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> Data:
        case_id = self.case_ids[index]
        group = self._h5()[f"data/{case_id}"]
        pos = torch.from_numpy(np.asarray(group["pos"], dtype=np.float32))
        pressure = torch.from_numpy(np.asarray(group["pressure"], dtype=np.float32)).reshape(-1, 1)
        target = self.normalize_pressure(pressure)
        sdf = torch.from_numpy(np.asarray(group["sdf"], dtype=np.float32))
        faces = torch.from_numpy(np.asarray(group["faces"], dtype=np.int64)).T.contiguous()
        return Data(
            x=torch.zeros((pos.shape[0], 1), dtype=torch.float32),
            y=target,
            pos=pos,
            pos_normalized=pos,
            latent_sdf=sdf.unsqueeze(0).unsqueeze(-1),
            face=faces,
            case_id=case_id,
            case_index=torch.tensor([int(case_id)], dtype=torch.long),
        )

    def de_normalize_pressure(self, value: torch.Tensor) -> torch.Tensor:
        return value * (self.pressure_std + self.pressure_eps) + self.pressure_mean

    def normalize_pressure(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.pressure_mean) / (self.pressure_std + self.pressure_eps)
