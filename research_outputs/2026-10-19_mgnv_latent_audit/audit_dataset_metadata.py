"""Read-only summary of the HDF5 files used by the ex3 latent-placement configs.

Run from the repository root. Only the JSON report beside this script is written;
no HDF5 content or model code is changed.
"""
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("R", "t", "L", "delta", "lam", "r_over_t", "l_over_r", "n_nodes")


def inspect(relative_path):
    with h5py.File(ROOT / relative_path, "r") as handle:
        rows = [dict(group.attrs) for group in handle["data"].values()]
    by_shape = {}
    for row in rows:
        by_shape.setdefault(int(row["shape_idx"]), {})[int(row["seed"])] = row
    representatives = [draws[min(draws)] for draws in by_shape.values()]
    summary = {
        "path": relative_path,
        "n_shapes": len(by_shape),
        "n_samples": len(rows),
        "draws_per_shape": sorted({len(draws) for draws in by_shape.values()}),
        "dimple_counts_K": sorted({int(row["K"]) for row in rows}),
        "tiers": sorted({str(row["tier"]) for row in rows}),
        "ranges": {field: [min(float(r[field]) for r in representatives),
                            max(float(r[field]) for r in representatives)]
                   for field in FIELDS},
        "unique_amplitude_mm_values": len({float(r["a_mm"]) for r in rows}),
        "unique_angle_values": len({float(r["theta_dimple"]) for r in rows}),
        "terminal_ke_ie_ratio": {
            "median": float(np.median([r["ke_ie_ratio"] for r in rows])),
            "max": float(max(r["ke_ie_ratio"] for r in rows)),
            "fraction_above_0_05": float(np.mean([r["ke_ie_ratio"] > 0.05 for r in rows])),
            "interpretation": "Descriptive metadata only; 0.05 is not an asserted solver acceptance criterion.",
        },
        "conditions": [{key: (int(r[key]) if key == "shape_idx" else
                                str(r[key]) if key == "tier" else float(r[key]))
                        for key in ("shape_idx", "tier", *FIELDS)}
                       for r in sorted(representatives, key=lambda r: int(r["shape_idx"]))],
    }
    return summary, by_shape


def main():
    train, train_shapes = inspect("dataset/probabilistic/ex3_shell_buckling.h5")
    infer, infer_shapes = inspect("dataset/probabilistic/ex3_shell_buckling_infer.h5")
    reference = train_shapes[min(train_shapes)]
    comparisons = []
    for split, shapes in (("train", train_shapes), ("infer", infer_shapes)):
        for shape_idx, draws in shapes.items():
            if set(draws) != set(reference):
                raise ValueError(f"Unexpected seed set: {split} shape {shape_idx}")
            comparisons.append({
                "split": split,
                "shape_idx": shape_idx,
                "max_abs_amplitude_difference": max(abs(float(draws[s]["a_mm"] - reference[s]["a_mm"])) for s in draws),
                "max_abs_angle_difference": max(abs(float(draws[s]["theta_dimple"] - reference[s]["theta_dimple"])) for s in draws),
                "max_abs_axial_recipe_quantile_difference": max(abs(
                    (float(draws[s]["z_dimple"]) - 2 * float(draws[s]["sigma"])) /
                    (float(draws[s]["L"]) - 4 * float(draws[s]["sigma"])) -
                    (float(reference[s]["z_dimple"]) - 2 * float(reference[s]["sigma"])) /
                    (float(reference[s]["L"]) - 4 * float(reference[s]["sigma"]))) for s in draws),
            })
    summary = {
        "client_date": "2026-10-19",
        "evidence": "Current HDF5 attributes; no field predictions or new FEA.",
        "datasets": {"train": train, "infer": infer},
        "common_recipe_comparison": {
            "reference_shape_idx": min(train_shapes),
            "n_shapes_checked": len(comparisons),
            "max_abs_amplitude_difference": max(r["max_abs_amplitude_difference"] for r in comparisons),
            "max_abs_angle_difference": max(r["max_abs_angle_difference"] for r in comparisons),
            "max_abs_axial_recipe_quantile_difference": max(r["max_abs_axial_recipe_quantile_difference"] for r in comparisons),
        },
    }
    output = Path(__file__).with_name("dataset_metadata_audit.json")
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output),
                      "common_recipes": summary["common_recipe_comparison"],
                      "counts": {name: {key: data[key] for key in ("n_shapes", "n_samples", "draws_per_shape", "dimple_counts_K")}
                                 for name, data in summary["datasets"].items()},
                      "terminal_ke_ie_ratio": {name: data["terminal_ke_ie_ratio"] for name, data in summary["datasets"].items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
