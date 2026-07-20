#!/usr/bin/env python3
"""Prepare the official Point-DeepONet benchmark without a 99 GiB download.

Kaggle exposes ``targets.npz`` and ``xyzdmlc.npz`` through its individual-file
download API.  Each NPZ is a ZIP whose members are one stored ``.npy`` array
per load case.  This module reads the ZIP central directories through HTTP byte
ranges, downloads only the selected case members, and writes one small,
resumable case file.  It never ranges into Kaggle's combined ``archive.zip``:
that outer bundle recompresses the NPZ files and cannot support nested random
access.

Normal Neural_Operator data loading and training are deliberately not imported
or modified here.  Running the script without ``--download`` only creates the
selection/provenance manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import struct
import time
import urllib.error
import urllib.request
import zlib
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


KAGGLE_DATASET_REF = "jangseop/point-deeponet-dataset"
KAGGLE_DATASET_VERSION = 2
KAGGLE_DOWNLOAD_API = (
    "https://api.kaggle.com/v1/datasets.DatasetApiService/DownloadDataset"
)
PREFIXES = ("ver", "hor", "dia")
EXCLUDED_ITEMS = (
    "102_240", "138_14", "149_256", "20_476", "200_610", "208_139",
    "225_120", "247_440", "249_321", "25_234", "280_556", "285_628",
    "293_424", "322_257", "348_556", "377_14", "377_82", "416_78",
    "421_466", "474_564", "486_507", "497_153", "506_25", "506_533",
    "527_8", "533_16", "536_282", "560_454", "56_324", "570_548",
    "623_129", "625_289", "72_209",
)
NPY_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
NPY_LOCAL_HEADER_SIGNATURE = 0x04034B50
SAMPLING_POLICY = "sha256-case-seed-v1"


@dataclass(frozen=True)
class SelectedCase:
    selection_rank: int
    case_name: str
    direction: str
    item_name: str
    mass_kg: float
    split: str
    split_rank: int


@dataclass(frozen=True)
class RemoteMember:
    archive_name: str
    member_name: str
    header_offset: int
    absolute_data_offset: int
    size: int
    crc32: int
    compression: int


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


def _resolve_redirect(request: urllib.request.Request, timeout: float = 30.0) -> str:
    """Resolve a short-lived redirect without beginning the target download."""
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        if error.code in {301, 302, 303, 307, 308} and error.headers.get("Location"):
            return error.headers["Location"]
        raise
    else:
        response.close()
        return response.geturl()


def resolve_kaggle_file_url(file_name: str, timeout: float = 30.0) -> str:
    """Resolve the official individual-file API to a signed GCS object URL."""
    payload = json.dumps(
        {
            "ownerSlug": "jangseop",
            "datasetSlug": "point-deeponet-dataset",
            "datasetVersionNumber": KAGGLE_DATASET_VERSION,
            "fileName": file_name,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        KAGGLE_DOWNLOAD_API,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "point-deeponet-preparer/1",
        },
    )
    return _resolve_redirect(request, timeout=timeout)


class HTTPRangeReader(io.RawIOBase):
    """Small seekable facade over an HTTP object with byte-range support."""

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 60.0,
        retries: int = 4,
        cache_block_size: int = 512 * 1024,
    ):
        super().__init__()
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self.cache_block_size = cache_block_size
        self._cache: dict[int, bytes] = {}
        self._position = 0
        self.size, self.etag, self.last_modified = self._probe()

    def _request(self, start: int, end: int) -> bytes:
        if start < 0 or end < start or end >= self.size:
            raise ValueError(f"invalid remote range {start}-{end} for {self.size} bytes")
        headers = {
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
            "User-Agent": "point-deeponet-preparer/1",
        }
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                request = urllib.request.Request(self.url, headers=headers)
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    if response.status != 206:
                        raise RuntimeError(
                            f"server ignored Range for {start}-{end}: HTTP {response.status}; "
                            "refusing a possible full-archive response"
                        )
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {start}-{end}/"):
                        raise RuntimeError(f"unexpected Content-Range: {content_range!r}")
                    data = response.read()
                expected = end - start + 1
                if len(data) != expected:
                    raise IOError(f"short range read: expected {expected}, received {len(data)}")
                return data
            except (OSError, RuntimeError, urllib.error.URLError) as error:
                last_error = error
                if attempt == self.retries:
                    break
                time.sleep(min(2 ** attempt, 8))
        assert last_error is not None
        raise last_error

    def _probe(self) -> tuple[int, str | None, str | None]:
        headers = {
            "Range": "bytes=0-0",
            "Accept-Encoding": "identity",
            "User-Agent": "point-deeponet-preparer/1",
        }
        request = urllib.request.Request(self.url, headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            if response.status != 206:
                raise RuntimeError(
                    f"remote object does not honor byte ranges (HTTP {response.status})"
                )
            content_range = response.headers.get("Content-Range", "")
            try:
                size = int(content_range.rsplit("/", 1)[1])
            except (IndexError, ValueError) as error:
                raise RuntimeError(f"invalid Content-Range probe: {content_range!r}") from error
            response.read(1)
            return size, response.headers.get("ETag"), response.headers.get("Last-Modified")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"invalid whence {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = min(position, self.size)
        return self._position

    def read(self, size: int = -1) -> bytes:
        if self._position >= self.size or size == 0:
            return b""
        if size is None or size < 0:
            size = self.size - self._position
        size = min(size, self.size - self._position)
        start = self._position
        if size > self.cache_block_size:
            data = self._request(start, start + size - 1)
        else:
            pieces = []
            remaining = size
            cursor = start
            while remaining:
                block_index = cursor // self.cache_block_size
                block_start = block_index * self.cache_block_size
                block = self._cache.get(block_index)
                if block is None:
                    block_end = min(block_start + self.cache_block_size, self.size) - 1
                    block = self._request(block_start, block_end)
                    # ZIP indexing needs only a few tail blocks. Keep memory bounded
                    # if a caller performs unrelated random reads later.
                    if len(self._cache) >= 4:
                        self._cache.pop(next(iter(self._cache)))
                    self._cache[block_index] = block
                within = cursor - block_start
                amount = min(remaining, len(block) - within)
                pieces.append(block[within : within + amount])
                cursor += amount
                remaining -= amount
            data = b"".join(pieces)
        self._position += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def _index_npz(
    remote: HTTPRangeReader,
    npz_basename: str,
) -> tuple[dict[str, RemoteMember], dict[str, int | str]]:
    with zipfile.ZipFile(remote) as npz:
        members: dict[str, RemoteMember] = {}
        for info in npz.infolist():
            if info.is_dir() or not info.filename.endswith(".npy"):
                continue
            key = Path(info.filename).name[:-4]
            if key in members:
                raise RuntimeError(f"duplicate NPZ key {key!r} in {npz_basename}")
            if info.compress_type != zipfile.ZIP_STORED:
                raise RuntimeError(
                    f"NPZ member {info.filename!r} is compressed; this preparer only "
                    "downloads stored members as exact byte ranges"
                )
            members[key] = RemoteMember(
                archive_name=npz_basename,
                member_name=info.filename,
                header_offset=info.header_offset,
                absolute_data_offset=-1,
                size=info.file_size,
                crc32=info.CRC,
                compression=info.compress_type,
            )
    metadata: dict[str, int | str] = {
        "file_name": npz_basename,
        "remote_size": remote.size,
        "remote_etag": remote.etag or "",
        "remote_last_modified": remote.last_modified or "",
        "npy_member_count": len(members),
    }
    return members, metadata


def _resolve_member_offset(remote: HTTPRangeReader, member: RemoteMember) -> RemoteMember:
    """Read one selected local header and return its exact payload range."""
    if member.absolute_data_offset >= 0:
        return member
    header = remote._request(
        member.header_offset,
        member.header_offset + NPY_LOCAL_HEADER.size - 1,
    )
    fields = NPY_LOCAL_HEADER.unpack(header)
    if fields[0] != NPY_LOCAL_HEADER_SIGNATURE:
        raise zipfile.BadZipFile(f"bad local header signature for {member.member_name!r}")
    filename_size, extra_size = fields[-2:]
    return RemoteMember(
        archive_name=member.archive_name,
        member_name=member.member_name,
        header_offset=member.header_offset,
        absolute_data_offset=(
            member.header_offset + NPY_LOCAL_HEADER.size + filename_size + extra_size
        ),
        size=member.size,
        crc32=member.crc32,
        compression=member.compression,
    )


class RemotePointDeepONetBundle:
    """Remote indexes for the two official NPZ files."""

    def __init__(
        self,
        *,
        targets_url: str | None = None,
        xyzdmlc_url: str | None = None,
    ):
        self.targets_source_url = targets_url or KAGGLE_DOWNLOAD_API
        self.xyzdmlc_source_url = xyzdmlc_url or KAGGLE_DOWNLOAD_API
        self.targets_resolved_url = targets_url or resolve_kaggle_file_url("targets.npz")
        self.xyzdmlc_resolved_url = xyzdmlc_url or resolve_kaggle_file_url("xyzdmlc.npz")
        self.targets_remote = HTTPRangeReader(self.targets_resolved_url)
        self.xyzdmlc_remote = HTTPRangeReader(self.xyzdmlc_resolved_url)
        self.targets, self.targets_metadata = _index_npz(self.targets_remote, "targets.npz")
        self.xyzdmlc, self.xyzdmlc_metadata = _index_npz(self.xyzdmlc_remote, "xyzdmlc.npz")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_digest(names: Iterable[str]) -> str:
    payload = "".join(f"{name}\n" for name in names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_author_selection(
    labels_csv: Path,
    *,
    n_samples: int = 1000,
    seed: int = 42,
) -> list[SelectedCase]:
    """Reproduce the released selection and mass-sorted 80/20 split cells."""
    item_mass: dict[str, float] = {}
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"item_name", "mass(kg)"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{labels_csv} must contain columns {sorted(required)}")
        for row in reader:
            item_name = row["item_name"]
            if item_name in EXCLUDED_ITEMS:
                continue
            if item_name in item_mass:
                raise ValueError(f"duplicate item_name {item_name!r} in {labels_csv}")
            item_mass[item_name] = float(row["mass(kg)"])

    candidates = [
        f"{prefix}_{item_name}"
        for item_name in sorted(item_mass)
        for prefix in PREFIXES
    ]
    if n_samples > len(candidates):
        raise ValueError(f"requested {n_samples} cases from only {len(candidates)} candidates")

    # The notebooks use np.random.seed/random.choice rather than default_rng.
    selection_rng = np.random.RandomState(seed)
    selected_names = selection_rng.choice(
        np.asarray(candidates, dtype=object), size=n_samples, replace=False
    ).tolist()
    selected_mass = {
        name: item_mass[name.split("_", 1)[1]] for name in selected_names
    }

    # np.savez preserves insertion order; dict(sorted(..., key=mass)) is stable.
    mass_sorted_names = [
        name for name, _ in sorted(selected_mass.items(), key=lambda item: item[1])
    ]
    split_rng = np.random.RandomState(seed)
    split_rng.shuffle(mass_sorted_names)
    train_count = int(n_samples * 0.8)
    split_by_name = {
        name: ("train" if rank < train_count else "valid", rank if rank < train_count else rank - train_count)
        for rank, name in enumerate(mass_sorted_names)
    }

    result = []
    for selection_rank, name in enumerate(selected_names):
        direction, item_name = name.split("_", 1)
        split, split_rank = split_by_name[name]
        result.append(
            SelectedCase(
                selection_rank=selection_rank,
                case_name=name,
                direction=direction,
                item_name=item_name,
                mass_kg=item_mass[item_name],
                split=split,
                split_rank=split_rank,
            )
        )
    return result


def _atomic_json(path: Path, data: Mapping | Sequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _write_manifests(
    output_dir: Path,
    cases: Sequence[SelectedCase],
    bundle: RemotePointDeepONetBundle,
    labels_csv: Path,
    *,
    n_points: int,
    selection_seed: int,
    sampling_seed: int,
) -> dict[str, object]:
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    ordered_by_split = sorted(cases, key=lambda case: (case.split != "train", case.split_rank))
    manifest_path = manifest_dir / "cases.csv"
    temporary = manifest_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(cases[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(case) for case in ordered_by_split)
    os.replace(temporary, manifest_path)

    for split in ("train", "valid"):
        names = [case.case_name for case in ordered_by_split if case.split == split]
        path = manifest_dir / f"{split}.txt"
        temporary = path.with_suffix(".txt.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.writelines(f"{name}\n" for name in names)
        os.replace(temporary, path)

    missing_targets = [case.case_name for case in cases if case.case_name not in bundle.targets]
    missing_inputs = [case.case_name for case in cases if case.case_name not in bundle.xyzdmlc]
    if missing_targets or missing_inputs:
        raise RuntimeError(
            f"selected archive members missing: targets={missing_targets[:5]}, "
            f"xyzdmlc={missing_inputs[:5]}"
        )

    selected_target_bytes = sum(bundle.targets[case.case_name].size for case in cases)
    selected_input_bytes = sum(bundle.xyzdmlc[case.case_name].size for case in cases)
    train_names = [case.case_name for case in ordered_by_split if case.split == "train"]
    valid_names = [case.case_name for case in ordered_by_split if case.split == "valid"]
    plan = {
        "case_count": len(cases),
        "train_count": len(train_names),
        "valid_count": len(valid_names),
        "unique_geometry_count": len({case.item_name for case in cases}),
        "selected_target_bytes": selected_target_bytes,
        "selected_xyzdmlc_bytes": selected_input_bytes,
        "selected_total_bytes": selected_target_bytes + selected_input_bytes,
        "train_manifest_sha256": manifest_digest(train_names),
        "valid_manifest_sha256": manifest_digest(valid_names),
    }
    _atomic_json(manifest_dir / "download_plan.json", plan)
    provenance = {
        "benchmark": "Point-DeepONet 1000-case paper subset",
        "kaggle_dataset_ref": KAGGLE_DATASET_REF,
        "kaggle_dataset_version": KAGGLE_DATASET_VERSION,
        "kaggle_download_endpoint": KAGGLE_DOWNLOAD_API,
        "targets_source_url": bundle.targets_source_url,
        "xyzdmlc_source_url": bundle.xyzdmlc_source_url,
        "labels_csv": str(labels_csv.resolve()),
        "labels_csv_sha256": sha256_file(labels_csv),
        "excluded_items": list(EXCLUDED_ITEMS),
        "prefix_order": list(PREFIXES),
        "selection_seed": selection_seed,
        "split_seed": selection_seed,
        "selection_protocol": (
            "sorted item_name; prefix order ver/hor/dia; NumPy RandomState.choice "
            "without replacement; stable mass sort; fresh RandomState.shuffle; 80/20"
        ),
        "n_points_per_case": n_points,
        "node_sampling_seed": sampling_seed,
        "node_sampling_policy": SAMPLING_POLICY,
        "node_sampling_note": (
            "The released notebook seeds one process-global RNG but iterates a Python set, "
            "so its per-case indices are not reproducible from source alone. This preparer "
            "uses an order-independent SHA-256-derived RandomState seed per case and retains "
            "the released choice-with/without-replacement rule."
        ),
        "targets_archive": bundle.targets_metadata,
        "xyzdmlc_archive": bundle.xyzdmlc_metadata,
        "output_schema": {
            "xyzdmlc": [n_points, 9],
            "targets": [n_points, 4],
            "sample_indices": [n_points],
            "dtype": "float32",
        },
        "plan": plan,
    }
    _atomic_json(manifest_dir / "provenance.json", provenance)
    return plan


def _case_seed(base_seed: int, case_name: str) -> int:
    digest = hashlib.sha256(f"{SAMPLING_POLICY}:{base_seed}:{case_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _crc32_file(path: Path) -> int:
    checksum = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


def _download_stored_member(
    remote: HTTPRangeReader,
    member: RemoteMember,
    destination: Path,
    *,
    chunk_size: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.exists() else 0
    if existing > member.size:
        raise RuntimeError(
            f"partial file {destination} is larger than its member ({existing} > {member.size})"
        )
    with destination.open("ab") as stream:
        while existing < member.size:
            amount = min(chunk_size, member.size - existing)
            start = member.absolute_data_offset + existing
            data = remote._request(start, start + amount - 1)
            stream.write(data)
            stream.flush()
            existing += len(data)
    checksum = _crc32_file(destination)
    if checksum != member.crc32:
        raise RuntimeError(
            f"CRC mismatch for {member.member_name}: {checksum:08x} != {member.crc32:08x}; "
            f"remove {destination} before retrying"
        )


def _validate_completed_case(path: Path, n_points: int, case_name: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                data["xyzdmlc"].shape == (n_points, 9)
                and data["targets"].shape == (n_points, 4)
                and data["sample_indices"].shape == (n_points,)
                and data["xyzdmlc"].dtype == np.float32
                and data["targets"].dtype == np.float32
                and str(data["case_name"].item()) == case_name
            )
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return False


def prepare_case(
    output_dir: Path,
    case: SelectedCase,
    bundle: RemotePointDeepONetBundle,
    *,
    n_points: int,
    sampling_seed: int,
    chunk_size: int,
) -> str:
    destination = output_dir / "cases" / case.split / f"{case.case_name}.npz"
    if _validate_completed_case(destination, n_points, case.case_name):
        return "skipped"

    stage = output_dir / ".staging" / case.case_name
    input_path = stage / "xyzdmlc.npy.part"
    target_path = stage / "targets.npy.part"
    input_member = _resolve_member_offset(
        bundle.xyzdmlc_remote, bundle.xyzdmlc[case.case_name]
    )
    target_member = _resolve_member_offset(
        bundle.targets_remote, bundle.targets[case.case_name]
    )
    _download_stored_member(
        bundle.xyzdmlc_remote,
        input_member,
        input_path,
        chunk_size=chunk_size,
    )
    _download_stored_member(
        bundle.targets_remote,
        target_member,
        target_path,
        chunk_size=chunk_size,
    )

    # A case is only a few MiB. Loading it eagerly avoids Windows mmap handles
    # keeping resumable staging files locked after the atomic output is written.
    inputs = np.load(input_path, allow_pickle=False)
    targets = np.load(target_path, allow_pickle=False)
    if inputs.ndim != 2 or inputs.shape[1] != 9 or inputs.dtype != np.float32:
        raise ValueError(f"{case.case_name} xyzdmlc must be float32 [N,9], got {inputs.shape} {inputs.dtype}")
    if targets.ndim != 2 or targets.shape[1] != 4 or targets.dtype != np.float32:
        raise ValueError(f"{case.case_name} targets must be float32 [N,4], got {targets.shape} {targets.dtype}")
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError(
            f"{case.case_name} row mismatch: xyzdmlc={inputs.shape[0]}, targets={targets.shape[0]}"
        )
    n_nodes = inputs.shape[0]
    rng = np.random.RandomState(_case_seed(sampling_seed, case.case_name))
    indices = rng.choice(n_nodes, n_points, replace=n_nodes <= n_points).astype(np.int32)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            xyzdmlc=np.asarray(inputs[indices], dtype=np.float32),
            targets=np.asarray(targets[indices], dtype=np.float32),
            sample_indices=indices,
            case_name=np.asarray(case.case_name),
            split=np.asarray(case.split),
            mass_kg=np.asarray(case.mass_kg, dtype=np.float32),
            original_node_count=np.asarray(n_nodes, dtype=np.int32),
            sampling_seed=np.asarray(_case_seed(sampling_seed, case.case_name), dtype=np.uint32),
        )
    os.replace(temporary, destination)

    # Staging files are recoverable remote-cache fragments and are needed only
    # until the atomic per-case output is complete.
    input_path.unlink()
    target_path.unlink()
    try:
        stage.rmdir()
    except OSError:
        pass
    return "written"


def _print_summary(plan: Mapping[str, object]) -> None:
    gib = 1024 ** 3
    print(
        f"Cases: {plan['case_count']} "
        f"({plan['train_count']} train / {plan['valid_count']} valid), "
        f"unique geometries: {plan['unique_geometry_count']}"
    )
    print(
        "Selected remote bytes: "
        f"targets={int(plan['selected_target_bytes']) / gib:.3f} GiB, "
        f"xyzdmlc={int(plan['selected_xyzdmlc_bytes']) / gib:.3f} GiB, "
        f"total={int(plan['selected_total_bytes']) / gib:.3f} GiB"
    )
    print(f"Train manifest SHA-256: {plan['train_manifest_sha256']}")
    print(f"Valid manifest SHA-256: {plan['valid_manifest_sha256']}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, default=root / "source" / "bracket_labels.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "prepared" / "n1000_p5000")
    parser.add_argument(
        "--targets-url",
        help="direct targets.npz URL; default resolves Kaggle's individual-file endpoint",
    )
    parser.add_argument(
        "--xyzdmlc-url",
        help="direct xyzdmlc.npz URL; default resolves Kaggle's individual-file endpoint",
    )
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-points", type=int, default=5000)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--chunk-mib", type=int, default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--download",
        action="store_true",
        help="download and prepare selected cases; omitted by default as a safety guard",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="index the remote ZIP and display case ranges without writing manifests/data",
    )
    parser.add_argument(
        "--limit-cases",
        type=int,
        help="process/print only the first N selected cases; use 1 for the one-case smoke path",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_samples <= 0 or args.n_points <= 0 or args.chunk_mib <= 0:
        raise ValueError("n-samples, n-points, and chunk-mib must be positive")
    cases = build_author_selection(
        args.labels_csv, n_samples=args.n_samples, seed=args.selection_seed
    )
    bundle = RemotePointDeepONetBundle(
        targets_url=args.targets_url,
        xyzdmlc_url=args.xyzdmlc_url,
    )

    missing_targets = [case.case_name for case in cases if case.case_name not in bundle.targets]
    missing_inputs = [case.case_name for case in cases if case.case_name not in bundle.xyzdmlc]
    if missing_targets or missing_inputs:
        raise RuntimeError(
            f"selected archive members missing: targets={missing_targets[:5]}, "
            f"xyzdmlc={missing_inputs[:5]}"
        )

    selected_target_bytes = sum(bundle.targets[case.case_name].size for case in cases)
    selected_input_bytes = sum(bundle.xyzdmlc[case.case_name].size for case in cases)
    train_names = [case.case_name for case in sorted(cases, key=lambda c: (c.split != "train", c.split_rank)) if case.split == "train"]
    valid_names = [case.case_name for case in sorted(cases, key=lambda c: (c.split != "train", c.split_rank)) if case.split == "valid"]
    plan = {
        "case_count": len(cases),
        "train_count": len(train_names),
        "valid_count": len(valid_names),
        "unique_geometry_count": len({case.item_name for case in cases}),
        "selected_target_bytes": selected_target_bytes,
        "selected_xyzdmlc_bytes": selected_input_bytes,
        "selected_total_bytes": selected_target_bytes + selected_input_bytes,
        "train_manifest_sha256": manifest_digest(train_names),
        "valid_manifest_sha256": manifest_digest(valid_names),
    }
    _print_summary(plan)

    limited_cases = cases[: args.limit_cases] if args.limit_cases else cases
    if args.dry_run:
        print(f"Dry run: showing {len(limited_cases)} case(s); no files will be written.")
        for case in limited_cases:
            input_member = _resolve_member_offset(
                bundle.xyzdmlc_remote, bundle.xyzdmlc[case.case_name]
            )
            target_member = _resolve_member_offset(
                bundle.targets_remote, bundle.targets[case.case_name]
            )
            print(
                f"{case.case_name} [{case.split}] "
                f"xyzdmlc=bytes {input_member.absolute_data_offset}-"
                f"{input_member.absolute_data_offset + input_member.size - 1}; "
                f"targets=bytes {target_member.absolute_data_offset}-"
                f"{target_member.absolute_data_offset + target_member.size - 1}"
            )
        return 0

    plan = _write_manifests(
        args.output_dir,
        cases,
        bundle,
        args.labels_csv,
        n_points=args.n_points,
        selection_seed=args.selection_seed,
        sampling_seed=args.sampling_seed,
    )
    print(f"Manifests written under {args.output_dir / 'manifests'}")
    if not args.download:
        print("Plan-only mode: pass --download to fetch case arrays.")
        return 0

    print(f"Preparing {len(limited_cases)} case(s) with resumable byte ranges.")
    written = 0
    skipped = 0
    for index, case in enumerate(limited_cases, start=1):
        status = prepare_case(
            args.output_dir,
            case,
            bundle,
            n_points=args.n_points,
            sampling_seed=args.sampling_seed,
            chunk_size=args.chunk_mib * 1024 * 1024,
        )
        written += status == "written"
        skipped += status == "skipped"
        print(f"[{index}/{len(limited_cases)}] {case.case_name}: {status}")
        _atomic_json(
            args.output_dir / "state.json",
            {
                "requested_this_run": len(limited_cases),
                "visited_this_run": index,
                "written_this_run": written,
                "skipped_this_run": skipped,
                "last_case": case.case_name,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
