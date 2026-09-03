import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cae_suite.specs import (
    build_chi_mgnflow_spec,
    build_meshgraphnets_spec,
    build_mlp_spec,
    build_neural_operator_spec,
    build_sdfflow_spec,
    build_simulgenvae_spec,
    build_transolver_spec,
    build_variational_spec,
)


def _catalogs():
    uri = (ROOT / "studio" / "src" / "constants.js").as_uri()
    script = f"import({uri!r}).then(m => console.log(JSON.stringify(m.KEY_CATALOGS)))"
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def _pairs():
    catalogs = _catalogs()
    return (
        ("mlp", catalogs["mlp"], build_mlp_spec()),
        ("meshgraphnets", catalogs["meshgraphnets"], build_meshgraphnets_spec()),
        ("meshgraphnetsV", catalogs["meshgraphnetsV"], build_variational_spec()),
        ("chiMgnflow", catalogs["chiMgnflow"], build_chi_mgnflow_spec()),
        ("operator", catalogs["operator"], build_neural_operator_spec()),
        ("transolver", catalogs["transolver"], build_transolver_spec()),
        ("sdfflow", catalogs["sdfflow"], build_sdfflow_spec()),
        ("simulgenvae", catalogs["simulgenvae"], build_simulgenvae_spec()),
    )


def test_studio_never_exposes_a_key_unknown_to_its_suite_spec():
    mismatches = {}
    for name, catalog, spec in _pairs():
        unknown = sorted(set(catalog) - set(spec.known_keys))
        if unknown:
            mismatches[name] = unknown
    assert not mismatches


def test_studio_catalog_contains_every_suite_spec_key():
    mismatches = {}
    for name, catalog, spec in _pairs():
        missing = sorted(set(spec.known_keys) - set(catalog))
        if missing:
            mismatches[name] = missing
    assert not mismatches


def test_studio_closed_choices_do_not_offer_values_rejected_by_native_runtime():
    uri = (ROOT / "studio" / "src" / "constants.js").as_uri()
    script = f"import({uri!r}).then(m => console.log(JSON.stringify(m.CHOICES)))"
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    choices = json.loads(completed.stdout)
    assert choices["coarsening_type"] == [
        "bfs", "voronoi_centroid", "voronoi_inherit", "voronoi_seedmean"
    ]
    assert choices["point_variant"] == ["mesh_state"]
    assert choices["point_sampling"] == ["random"]
    assert choices["out_of_bounds_policy"] == ["error", "clamp"]
    assert choices["sdf_source"] == ["none", "dataset", "sidecar"]

    parallel_script = (
        f"import({uri!r}).then(m => "
        "console.log(JSON.stringify(m.PARALLEL_MODE_CHOICES)))"
    )
    parallel = subprocess.run(
        ["node", "--input-type=module", "-e", parallel_script],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    parallel_choices = json.loads(parallel.stdout)
    assert parallel_choices["chi-mgnflow"] == ["ddp"]


if __name__ == "__main__":
    for name, catalog, spec in _pairs():
        gui_only = sorted(set(catalog) - set(spec.known_keys))
        suite_only = sorted(set(spec.known_keys) - set(catalog))
        print(f"{name}: gui-only={len(gui_only)} {' '.join(gui_only)}")
        print(f"{name}: suite-only={len(suite_only)} {' '.join(suite_only)}")
