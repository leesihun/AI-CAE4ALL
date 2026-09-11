"""Reproduce mathematical checks accompanying the enhancement research.

Standard library only. No training, checkpoints, GPU, or dataset mutation.
These calculations establish mathematical distinctions, not model performance.
Run from any directory: python <path-to-this-file>
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def midpoint_integral(function, n=100_000):
    return math.fsum(function((i + 0.5) / n) for i in range(n)) / n


def main():
    # Independent source Z0~N(0,1), trainable target Z1~N(0,a^2).
    # At the optimal CFM regressor the residual is Var(U | Zt).
    # Its integral is pi*a/2, although a matching prior has KL=0 for every a.
    fm_rows = []
    for a in (0.25, 0.5, 1.0, 2.0):
        numerical = midpoint_integral(
            lambda t: a * a / ((1 - t) ** 2 + a * a * t * t)
        )
        expected = math.pi * a / 2
        assert math.isclose(numerical, expected, rel_tol=1e-8)
        fm_rows.append({"target_std": a, "optimal_cfm_mse": numerical,
                        "analytic_cfm_mse": expected, "matching_prior_kl": 0.0})

    # Expected negatively-oriented CRPS for forecast N(0,b^2), truth N(0,1).
    # One observation is scored per independent verification case.
    crps_rows = []
    for b in (0.0, 0.5, 1.0, 2.0):
        score = math.sqrt(2 / math.pi) * math.sqrt(1 + b * b) - b / math.sqrt(math.pi)
        crps_rows.append({"forecast_std": b, "expected_crps": score})
    assert min(crps_rows, key=lambda row: row["expected_crps"])["forecast_std"] == 1.0

    sigma_min = 1e-4
    s = 1 - sigma_min
    readout_errors = []
    residual_variance_errors = []
    for t in (0.0, 0.1, 0.5, 0.9, 1.0):
        a = 1 - s * t
        for z0, y1 in ((-2.0, 3.0), (0.25, -1.5)):
            yt = a * z0 + t * y1
            u = y1 - s * z0
            readout_errors.append(abs(s * yt + a * u - y1))
        for target_var in (0.25, 1.0, 4.0):
            var_yt = a * a + t * t * target_var
            covariance = t * target_var - s * a
            residual_var = target_var + s * s - covariance * covariance / var_yt
            residual_variance_errors.append(abs(residual_var - target_var / var_yt))
    assert max(readout_errors) < 1e-12
    assert max(residual_variance_errors) < 1e-12

    # The current self-normalized time weight cancels with one graph per batch.
    node_errors = (1.0, 2.0, 4.0)
    weighted_rows = []
    for weight in (1.0, 0.25, 0.01):
        reduced = sum(weight * value for value in node_errors) / (weight * len(node_errors))
        weighted_rows.append({"time_weight": weight, "self_normalized_loss": reduced})
    assert max(row["self_normalized_loss"] for row in weighted_rows) - min(
        row["self_normalized_loss"] for row in weighted_rows) < 1e-12

    checks = {
        "scope": "Analytical examples; no empirical claims about trained models or SAOI",
        "cfm_is_not_kl": fm_rows,
        "crps_with_one_truth_per_case": crps_rows,
        "clean_readout_max_abs_error": max(readout_errors),
        "proposed_gaussian_preconditioner_variance_max_abs_error": max(residual_variance_errors),
        "self_normalized_time_weight_single_graph": weighted_rows,
        "iid_unit_noise_cluster_mean_variance": {str(n): 1 / n for n in (1, 10, 100, 1000)},
        "nonlinear_decoder_inflation": {
            "setup": "Z~N(0,1), D(z)=z^2; latent mean remains zero",
            "mean_D_Z": 1.0, "mean_D_2Z": 4.0,
            "meaning": "Even mean-preserving latent inflation need not preserve the decoded field mean",
        },
        "largest_time_embedding_frequency_cycles_over_unit_interval": 2 ** 14,
    }
    (HERE / "reasoning_checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")

    paths = [
        "methods/MeshGraphNets_Variational/model/MeshGraphNets.py",
        "methods/MeshGraphNets_Variational/model/vae.py",
        "methods/MeshGraphNets_Variational/model/conditional_prior.py",
        "methods/MeshGraphNets_Variational/training_profiles/training_loop.py",
        "methods/MeshGraphNets_Variational/inference_profiles/rollout.py",
        "methods/HI_MGNFlow/model/flow.py",
        "methods/HI_MGNFlow/model/CHiMGNFlow.py",
        "methods/HI_MGNFlow/model/coarsening.py",
        "methods/HI_MGNFlow/training_profiles/training_loop.py",
        "methods/HI_MGNFlow/inference_profiles/rollout.py",
        "methods/HI_MGNFlow/general_modules/mesh_dataset.py",
        "configs/MeshGraphNets_Variational/SAOI_all_input/config_train_bot.txt",
        "configs/HI_MGNFlow/SAOI_all_input/config_train_bot.txt",
        "configs/MeshGraphNets_Variational/SAOI_all_input/config_infer_s26fe_main_bot.txt",
        "configs/HI_MGNFlow/SAOI_all_input/config_infer_s26fe_main_bot.txt",
        "docs/research/SAOI_PROBABILISTIC_SWEEP_2026-09.md",
        "output/meshgraphnets-v/saoi_sweep3/sweep_results.md",
    ]
    absent_paths = [
        "dataset/SAOI",
        "output/meshgraphnets-v/saoi_sweep3/infer",
        "output/chi-mgnflow/saoi_sweepB/infer",
    ]
    snapshot = {
        "review_date": "2026-09-11",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files": [{"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
                   "lines": len((ROOT / path).read_text(encoding="utf-8").splitlines())} for path in paths],
        "data_and_dump_availability": {path: (ROOT / path).exists() for path in absent_paths},
        "method_output_artifacts": {
            directory: sorted(str(path.relative_to(ROOT)).replace("\\", "/")
                              for path in (ROOT / directory).rglob("*") if path.is_file())
            for directory in ("output/meshgraphnets-v", "output/chi-mgnflow")
        },
    }
    (HERE / "audit_snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
