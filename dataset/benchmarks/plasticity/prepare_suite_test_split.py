#!/usr/bin/env python3
"""Materialize the suite's deterministic held-out Plasticity cases.

All deterministic temporal backends currently split the source sample IDs with
``numpy.random.default_rng(seed)`` and an 80/10/10 partition.  This helper
copies exactly the test IDs into a standalone HDF5 file so rollout inference
does not evaluate training or validation cases.  It changes no model code and
keeps the original sample IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import h5py
import numpy as np


TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=here / "plasticity.h5")
    parser.add_argument(
        "--output", type=Path, default=here / "plasticity_seed42_test.h5"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def suite_split(sample_ids: list[int], seed: int) -> tuple[np.ndarray, ...]:
    shuffled = np.asarray(sample_ids, dtype=np.int64)
    np.random.default_rng(seed).shuffle(shuffled)
    n_train = int(len(shuffled) * TRAIN_RATIO)
    n_val = int(len(shuffled) * VAL_RATIO)
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


def copy_test_file(source: Path, temporary: Path, seed: int) -> tuple[np.ndarray, ...]:
    with h5py.File(source, "r") as src:
        sample_ids = sorted(int(key) for key in src["data"].keys())
        train_ids, val_ids, test_ids = suite_split(sample_ids, seed)

        with h5py.File(temporary, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            dst.attrs["num_samples"] = len(test_ids)
            dst.attrs["suite_split_seed"] = seed
            dst.attrs["suite_split_role"] = "test"
            dst.attrs["suite_split_source"] = source.name
            dst.attrs["suite_split_train_count"] = len(train_ids)
            dst.attrs["suite_split_val_count"] = len(val_ids)
            dst.attrs["suite_split_test_count"] = len(test_ids)

            src.copy("topology", dst)
            metadata = dst.create_group("metadata")
            for name in src["metadata"]:
                if name != "splits":
                    src.copy(src[f"metadata/{name}"], metadata, name=name)

            splits = dst.create_group("splits")
            splits.create_dataset("train", data=np.empty(0, dtype=np.int64))
            splits.create_dataset("val", data=np.empty(0, dtype=np.int64))
            splits.create_dataset("test", data=np.sort(test_ids))
            splits.create_dataset("unused", data=np.empty(0, dtype=np.int64))
            metadata["splits"] = splits

            data = dst.create_group("data")
            for sample_id in sorted(int(value) for value in test_ids):
                src_group = src[f"data/{sample_id}"]
                dst_group = data.create_group(str(sample_id))
                src.copy(src_group["nodal_data"], dst_group, name="nodal_data")
                dst_group["mesh_edge"] = dst["topology/mesh_edge_structured_quad"]
                src.copy(src_group["die_profile"], dst_group, name="die_profile")
                src.copy(src_group["metadata"], dst_group, name="metadata")

    return train_ids, val_ids, test_ids


def audit(path: Path, expected_test_ids: np.ndarray, seed: int) -> None:
    expected = sorted(int(value) for value in expected_test_ids)
    with h5py.File(path, "r") as handle:
        actual = sorted(int(key) for key in handle["data"].keys())
        if actual != expected:
            raise ValueError("Materialized test sample IDs do not match the seeded split")
        if handle.attrs["suite_split_seed"] != seed:
            raise ValueError("Stored split seed mismatch")
        if list(handle["splits/test"][:]) != expected:
            raise ValueError("splits/test is not the sorted materialized ID set")
        for sample_id in actual:
            group = handle[f"data/{sample_id}"]
            if group["nodal_data"].shape != (8, 20, 3131):
                raise ValueError(f"Sample {sample_id}: unexpected nodal_data shape")
            if group["mesh_edge"].shape != (2, 6130):
                raise ValueError(f"Sample {sample_id}: unexpected mesh_edge shape")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Missing source HDF5 file: {source}")
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)

    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        train_ids, val_ids, test_ids = copy_test_file(source, temporary, args.seed)
        audit(temporary, test_ids, args.seed)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"Wrote {output} ({output.stat().st_size / 1024**2:.1f} MiB)")
    print(
        f"Seed {args.seed} split counts: train={len(train_ids)}, "
        f"val={len(val_ids)}, test={len(test_ids)}"
    )
    print(f"Test IDs: {','.join(str(value) for value in sorted(test_ids))}")
    print(f"SHA256: {sha256(output)}")


if __name__ == "__main__":
    main()
