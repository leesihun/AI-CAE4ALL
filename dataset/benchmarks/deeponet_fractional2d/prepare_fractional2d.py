#!/usr/bin/env python3
"""Generate the original DeepONet paper's 2D fractional-Laplacian data.

The authors released MATLAB generation code but no pre-generated data.  This
is a direct NumPy/SciPy translation of ``fractional/Fractional_Lap_2D.m`` and
its helpers.  Data stay compact: branch vectors and operator targets are
stored once, while the released train/test groups are HDF5 hard links because
the released 2D MATLAB functions instantiate the same Sobol stream in both.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np
from scipy import special


PAPER_URL = "https://doi.org/10.1038/s42256-021-00302-5"
OFFICIAL_REPOSITORY = "https://github.com/lululxvi/deeponet"
OFFICIAL_COMMIT = "8d62345afd39e1df9c2c8c8d0e7c41882b06a9bf"
MATLAB_SOBOL_DOCUMENTATION = "https://www.mathworks.com/help/stats/sobolset.html"
DIRECTION_FILE_URL = "https://web.maths.unsw.edu.au/~fkuo/sobol/joe-kuo-old.1111"
PAPER_PLOT_NORMALIZED_MSE = 1.2e-3  # approximate reading of Figure 2e, not a table value
DIRECTION_FILE_SHA256 = "864a98b3af71806c1922feed53b9f77da29189f67a52bba0a88f7503d332e949"


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=here / "fractional2d_released.h5")
    parser.add_argument(
        "--direction-file", type=Path,
        default=here / "source" / "joe-kuo-old.1111",
        help="Joe-Kuo 2003 direction table used by MATLAB sobolset/qrandstream.",
    )
    parser.add_argument("--num-functions", type=int, default=5000)
    parser.add_argument("--num-grid", type=int, default=15)
    parser.add_argument("--num-alpha", type=int, default=10)
    parser.add_argument("--max-zernike-index", type=int, default=14)
    parser.add_argument("--coefficient-half-width", type=float, default=2.0)
    parser.add_argument("--step", type=float, default=1.0e-3)
    parser.add_argument("--angular-points", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-analytic-check", action="store_true",
        help="Skip the released alpha=1.5 manufactured-solution audit.",
    )
    return parser.parse_args()


def zernike15(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """The normalized Z0..Z14 formulas from the released ``Zp.m``."""
    r = np.asarray(r, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    out = np.empty(r.shape + (15,), dtype=np.float64)
    out[..., 0] = 1.0
    out[..., 1] = 2.0 * r * np.sin(theta)
    out[..., 2] = 2.0 * r * np.cos(theta)
    out[..., 3] = math.sqrt(6.0) * r**2 * np.sin(2.0 * theta)
    out[..., 4] = math.sqrt(3.0) * (2.0 * r**2 - 1.0)
    out[..., 5] = math.sqrt(6.0) * r**2 * np.cos(2.0 * theta)
    out[..., 6] = math.sqrt(8.0) * r**3 * np.sin(3.0 * theta)
    out[..., 7] = math.sqrt(8.0) * (3.0 * r**3 - 2.0 * r) * np.sin(theta)
    out[..., 8] = math.sqrt(8.0) * (3.0 * r**3 - 2.0 * r) * np.cos(theta)
    out[..., 9] = math.sqrt(8.0) * r**3 * np.cos(3.0 * theta)
    out[..., 10] = math.sqrt(10.0) * r**4 * np.sin(4.0 * theta)
    out[..., 11] = math.sqrt(10.0) * (4.0 * r**4 - 3.0 * r**2) * np.sin(2.0 * theta)
    out[..., 12] = math.sqrt(5.0) * (6.0 * r**4 - 6.0 * r**2 + 1.0)
    out[..., 13] = math.sqrt(10.0) * (4.0 * r**4 - 3.0 * r**2) * np.cos(2.0 * theta)
    out[..., 14] = math.sqrt(10.0) * r**4 * np.cos(4.0 * theta)
    return out


def matlab_polar_grid(num_grid: int, max_radius: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Return MATLAB ``meshgrid(...); array(:)`` ordering."""
    radii = np.linspace(0.0, max_radius, num_grid, dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, num_grid, dtype=np.float64)
    radial_grid, angular_grid = np.meshgrid(radii, angles, indexing="xy")
    polar = np.column_stack((radial_grid.ravel(order="F"), angular_grid.ravel(order="F")))
    xy = np.column_stack((
        polar[:, 0] * np.cos(polar[:, 1]),
        polar[:, 0] * np.sin(polar[:, 1]),
    ))
    return polar, xy


def matlab_standard_sobol(
    logical_indices: np.ndarray,
    dimension: int,
    direction_file: Path,
    bits: int = 32,
) -> np.ndarray:
    """Generate MATLAB ``PointOrder='standard'`` Sobol points by index."""
    direction_file = Path(direction_file)
    actual_hash = sha256(direction_file)
    if actual_hash != DIRECTION_FILE_SHA256:
        raise ValueError(
            f"Direction table SHA-256 mismatch for {direction_file}: {actual_hash}"
        )

    table: dict[int, tuple[int, int, list[int]]] = {}
    with direction_file.open("r", encoding="ascii") as handle:
        next(handle)
        for raw_line in handle:
            fields = [int(value) for value in raw_line.split()]
            if not fields:
                continue
            dim, degree, coefficient, *initial = fields
            if len(initial) != degree:
                raise ValueError(f"Malformed direction row for dimension {dim}")
            table[dim] = (degree, coefficient, initial)

    directions = np.zeros((dimension, bits), dtype=np.uint32)
    for bit in range(bits):
        directions[0, bit] = np.uint32(1 << (bits - bit - 1))

    for dim in range(2, dimension + 1):
        if dim not in table:
            raise ValueError(f"Direction table lacks required dimension {dim}")
        degree, coefficient, initial = table[dim]
        for bit in range(degree):
            directions[dim - 1, bit] = np.uint32(
                initial[bit] << (bits - bit - 1)
            )
        for bit in range(degree, bits):
            value = directions[dim - 1, bit - degree]
            value ^= np.right_shift(value, degree)
            for offset in range(1, degree):
                coefficient_bit = (coefficient >> (degree - 1 - offset)) & 1
                if coefficient_bit:
                    value ^= directions[dim - 1, bit - offset]
            directions[dim - 1, bit] = value

    logical_indices = np.asarray(logical_indices, dtype=np.uint64)
    if logical_indices.size and int(logical_indices.max()).bit_length() > bits:
        raise ValueError(f"Sobol index exceeds configured {bits}-bit generator")
    integer_points = np.zeros((len(logical_indices), dimension), dtype=np.uint32)
    for bit in range(bits):
        selected = ((logical_indices >> bit) & 1).astype(bool)
        if np.any(selected):
            integer_points[selected] ^= directions[:, bit]
    return integer_points.astype(np.float64) / float(1 << bits)


def released_sobol_coefficients(
    num_functions: int,
    dimension: int = 15,
    half_width: float = 2.0,
    direction_file: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate qrandstream Sobol Skip=3, Leap=24, then drop temp row 1.

    MATLAB's default is ``PointOrder='standard'`` and uses the Joe-Kuo 2003
    primitive polynomials/direction numbers. The official direction table is
    consumed directly instead of relying on SciPy's newer 2008 table.
    """
    skip = 3
    leap = 24
    logical_indices = skip + np.arange(num_functions + 1, dtype=np.int64) * (leap + 1)
    if direction_file is None:
        direction_file = Path(__file__).resolve().parent / "source" / "joe-kuo-old.1111"
    selected = matlab_standard_sobol(
        logical_indices, dimension, Path(direction_file)
    )[1:]
    coefficients = -half_width + 2.0 * half_width * selected
    return coefficients.astype(np.float64, copy=False), logical_indices[1:]


def grunwald_weights(alpha: float, max_steps: int) -> np.ndarray:
    """Stable recurrence equivalent to the released product in ``wk``."""
    weights = np.empty(max_steps + 1, dtype=np.float64)
    weights[0] = 1.0
    for index in range(1, max_steps + 1):
        weights[index] = weights[index - 1] * ((index - 1.0 - alpha) / index)
    return weights


def fractional_operator_on_zernike(
    query_xy: np.ndarray,
    alpha_values: np.ndarray,
    *,
    step: float = 1.0e-3,
    angular_points: int = 16,
    max_steps: int = 8000,
) -> np.ndarray:
    """Vector Grünwald-Letnikov operator applied to the 15 basis fields."""
    query_xy = np.asarray(query_xy, dtype=np.float64)
    alpha_values = np.asarray(alpha_values, dtype=np.float64)
    nodes, legendre_weights = np.polynomial.legendre.leggauss(angular_points)
    directions = np.pi + np.pi * nodes
    angular_weights = np.pi * legendre_weights

    gk = np.stack([grunwald_weights(float(alpha), max_steps) for alpha in alpha_values])
    ratios = (
        special.gamma((1.0 - alpha_values) / 2.0)
        * special.gamma((2.0 + alpha_values) / 2.0)
        / math.sqrt(math.pi)
        / 2.0
        / math.pi
    )
    scales = ratios * step ** (-alpha_values)
    indices = np.arange(max_steps + 1, dtype=np.float64)
    result = np.zeros((len(alpha_values), len(query_xy), 15), dtype=np.float64)

    for query_index, (x0, y0) in enumerate(query_xy):
        for direction, angular_weight in zip(directions, angular_weights):
            x_path = x0 - indices * step * math.cos(float(direction))
            y_path = y0 - indices * step * math.sin(float(direction))
            outside = np.flatnonzero(x_path * x_path + y_path * y_path > 1.0)
            if outside.size == 0:
                raise ValueError(
                    f"max_steps={max_steps} is too small for query {query_index}"
                )
            path_length = int(outside[0])
            if path_length == 0:
                continue
            radius = np.hypot(x_path[:path_length], y_path[:path_length])
            theta = np.arctan2(y_path[:path_length], x_path[:path_length])
            basis_on_path = zernike15(radius, theta)
            result[:, query_index, :] += (
                angular_weight * (gk[:, :path_length] @ basis_on_path)
            )

    return result * scales[:, None, None]


def manufactured_solution_audit(
    sensor_polar: np.ndarray,
    query_xy: np.ndarray,
    *,
    step: float,
    angular_points: int,
    max_steps: int,
) -> dict[str, float]:
    """Reproduce the alpha=1.5 check embedded in released ``test_set.m``."""
    alpha = 1.5
    dense_polar, _ = matlab_polar_grid(60, max_radius=1.0)
    dense_basis = zernike15(dense_polar[:, 0], dense_polar[:, 1])
    dense_exact = (1.0 - dense_polar[:, 0] ** 2) ** (1.0 + alpha / 2.0)
    coefficients, *_ = np.linalg.lstsq(dense_basis, dense_exact, rcond=None)
    projection = dense_basis @ coefficients
    projection_relative_l2 = float(
        np.linalg.norm(projection - dense_exact) / np.linalg.norm(dense_exact)
    )

    operator_basis = fractional_operator_on_zernike(
        query_xy,
        np.asarray([alpha]),
        step=step,
        angular_points=angular_points,
        max_steps=max_steps,
    )[0]
    approximate = operator_basis @ coefficients
    radius_squared = np.sum(query_xy**2, axis=1)
    exact = (
        2.0**alpha
        * special.gamma(alpha / 2.0 + 2.0)
        * special.gamma((2.0 + alpha) / 2.0)
        * (1.0 - (1.0 + alpha / 2.0) * radius_squared)
    )
    operator_relative_l2 = float(
        np.linalg.norm(approximate - exact) / np.linalg.norm(exact)
    )

    # ``u0`` is not needed for training, but this catches sensor ordering.
    sensor_basis = zernike15(sensor_polar[:, 0], sensor_polar[:, 1])
    sensor_norm = float(np.linalg.norm(sensor_basis @ coefficients))
    return {
        "projection_relative_l2": projection_relative_l2,
        "operator_relative_l2": operator_relative_l2,
        "manufactured_sensor_norm": sensor_norm,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataset(args: argparse.Namespace) -> dict[str, object]:
    if args.max_zernike_index != 14:
        raise ValueError("The paper's 2D benchmark requires max_zernike_index=14")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} exists; pass --force to replace it")

    sensor_polar, query_xy = matlab_polar_grid(args.num_grid)
    alpha_values = np.linspace(0.01, 0.99, args.num_alpha, dtype=np.float64)
    sensor_basis = zernike15(sensor_polar[:, 0], sensor_polar[:, 1])
    coefficients, sobol_logical_indices = released_sobol_coefficients(
        args.num_functions, 15, args.coefficient_half_width, args.direction_file
    )
    operator_basis = fractional_operator_on_zernike(
        query_xy,
        alpha_values,
        step=args.step,
        angular_points=args.angular_points,
        max_steps=args.max_steps,
    )
    branch_values = coefficients @ sensor_basis.T
    targets = np.einsum("fk,aqk->faq", coefficients, operator_basis, optimize=True)

    audit: dict[str, float] = {}
    if not args.skip_analytic_check and args.num_grid == 15:
        audit = manufactured_solution_audit(
            sensor_polar,
            query_xy,
            step=args.step,
            angular_points=args.angular_points,
            max_steps=args.max_steps,
        )
        if audit["projection_relative_l2"] > 0.01:
            raise ValueError(f"Zernike projection audit failed: {audit}")
        if audit["operator_relative_l2"] > 0.06:
            raise ValueError(f"Fractional-operator audit failed: {audit}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        handle.attrs["format"] = "deeponet_fractional_laplacian_2d_compact_v1"
        handle.attrs["paper_url"] = PAPER_URL
        handle.attrs["official_repository"] = OFFICIAL_REPOSITORY
        handle.attrs["official_commit"] = OFFICIAL_COMMIT
        handle.attrs["released_train_test_relation"] = "identical_hard_links"
        handle.attrs["paper_result_kind"] = "plot_derived_approximate_normalized_mse"
        handle.attrs["paper_result_approximate"] = PAPER_PLOT_NORMALIZED_MSE
        handle.attrs["sobol_qualification"] = (
            "MATLAB standard point order; Joe-Kuo 2003 joe-kuo-old.1111 direction table; "
            "Skip=3 Leap=24; temp first row dropped"
        )
        handle.attrs["sobol_direction_file_sha256"] = DIRECTION_FILE_SHA256
        handle.attrs["sobol_documentation"] = MATLAB_SOBOL_DOCUMENTATION
        handle.attrs["sobol_direction_file_url"] = DIRECTION_FILE_URL
        handle.attrs["manufactured_solution_audit"] = json.dumps(audit, sort_keys=True)

        common = handle.create_group("common")
        common.create_dataset("sensor_polar", data=sensor_polar.astype(np.float32))
        common.create_dataset("query_xy", data=query_xy.astype(np.float32))
        common.create_dataset("alpha", data=alpha_values.astype(np.float32))
        common.create_dataset("sobol_logical_index", data=sobol_logical_indices)
        common.create_dataset("operator_basis", data=operator_basis.astype(np.float32))

        arrays = handle.create_group("arrays")
        arrays.create_dataset(
            "coefficients", data=coefficients.astype(np.float32), compression="gzip", shuffle=True
        )
        arrays.create_dataset(
            "branch_values", data=branch_values.astype(np.float32),
            chunks=(min(256, args.num_functions), branch_values.shape[1]),
            compression="gzip", shuffle=True,
        )
        arrays.create_dataset(
            "targets", data=targets.astype(np.float32),
            chunks=(1, args.num_alpha, query_xy.shape[0]),
            compression="gzip", shuffle=True,
        )

        for split_name in ("train", "test"):
            split = handle.create_group(split_name)
            split["coefficients"] = arrays["coefficients"]
            split["branch_values"] = arrays["branch_values"]
            split["targets"] = arrays["targets"]
            split.attrs["num_functions"] = args.num_functions
        handle.flush()

    result: dict[str, object] = {
        "output": str(args.output.resolve()),
        "sha256": sha256(args.output),
        "num_functions_per_released_split": args.num_functions,
        "branch_shape": list(branch_values.shape),
        "target_shape": list(targets.shape),
        "expanded_triples_per_split": int(targets.size),
        "released_test_duplicates_train": True,
        "manufactured_solution_audit": audit,
    }
    return result


def main() -> None:
    result = write_dataset(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
