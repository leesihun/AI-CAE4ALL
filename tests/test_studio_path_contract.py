import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _studio_paths() -> dict[str, str]:
    autofill_uri = (ROOT / "studio" / "src" / "autofill.js").as_uri()
    constants_uri = (ROOT / "studio" / "src" / "constants.js").as_uri()
    script = f"""
      const autofill = await import({autofill_uri!r});
      const constants = await import({constants_uri!r});
      console.log(JSON.stringify({{
        dataset: autofill.toMethodPath('dataset/ex9.h5'),
        output: autofill.toMethodPath('output/example/model.pth'),
        absolute: autofill.toMethodPath('C:/data/example.h5'),
        alreadyRelative: autofill.toMethodPath('../../dataset/ex9.h5'),
        geometryInput: autofill.toGeometryPath('dataset/part.stl'),
        geometryOutput: autofill.fromGeometryPath('../../studio/runtime/part.h5'),
        ex9Dataset: constants.TEMPLATES.himgn.nodes.find(row => row[0] === 'trainer')[4].dataset_dir,
        ex9Inference: constants.TEMPLATES.himgn.nodes.find(row => row[0] === 'trainer')[4].infer_dataset
      }}));
    """
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def test_studio_suite_paths_are_relative_to_nested_method_repositories() -> None:
    paths = _studio_paths()
    assert paths["dataset"] == "../../dataset/ex9.h5"
    assert paths["output"] == "../../output/example/model.pth"
    assert paths["absolute"] == "C:/data/example.h5"
    assert paths["alreadyRelative"] == "../../dataset/ex9.h5"
    assert paths["geometryInput"] == "../../dataset/part.stl"
    assert paths["geometryOutput"] == "studio/runtime/part.h5"
    assert paths["ex9Dataset"] == "../../dataset/ex9.h5"
    assert paths["ex9Inference"] == "../../dataset/ex9_infer.h5"

    repository = ROOT / "methods" / "MeshGraphNets"
    assert (repository / paths["dataset"]).resolve() == (ROOT / "dataset" / "ex9.h5").resolve()
