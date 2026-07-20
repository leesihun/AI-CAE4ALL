from __future__ import annotations

import importlib.util
import io
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "prepare_dataset.py"
SPEC = importlib.util.spec_from_file_location("point_deeponet_prepare_dataset", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _npz_bytes(case_name: str, array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.savez(stream, **{case_name: array})
    return stream.getvalue()


class _RangeHandler(BaseHTTPRequestHandler):
    payloads: dict[str, bytes] = {}
    requests: list[tuple[str, int, int]] = []

    def do_GET(self):  # noqa: N802
        payload = self.payloads.get(self.path)
        if payload is None:
            self.send_error(404)
            return
        range_header = self.headers.get("Range")
        if not range_header or not range_header.startswith("bytes="):
            self.send_error(416, "range required by test server")
            return
        start_text, end_text = range_header[6:].split("-", 1)
        start = int(start_text)
        end = min(int(end_text), len(payload) - 1)
        self.requests.append((self.path, start, end))
        body = payload[start : end + 1]
        self.send_response(206)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("ETag", '"synthetic"')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        return


class PreparePointDeepONetTests(unittest.TestCase):
    def test_author_1000_selection_regression(self):
        labels = MODULE_PATH.parent / "source" / "bracket_labels.csv"
        cases = MODULE.build_author_selection(labels, n_samples=1000, seed=42)
        ordered = sorted(cases, key=lambda case: (case.split != "train", case.split_rank))
        train = [case.case_name for case in ordered if case.split == "train"]
        valid = [case.case_name for case in ordered if case.split == "valid"]

        self.assertEqual((len(train), len(valid)), (800, 200))
        self.assertEqual(len({case.item_name for case in cases}), 837)
        self.assertFalse({case.item_name for case in cases}.intersection(MODULE.EXCLUDED_ITEMS))
        self.assertEqual(
            MODULE.manifest_digest(train),
            "b1153fad047e45bfe5bbdda15cb93ecb6e30983de4bdd7824a9111a044c88d33",
        )
        self.assertEqual(
            MODULE.manifest_digest(valid),
            "282cab5b4e3ec9dd45f8f59399a9bdda06bf99165c848cbf443fc66bd419b05a",
        )

    def test_one_case_range_extract_crc_and_resume(self):
        case_name = "ver_1_1"
        xyzdmlc = np.arange(7 * 9, dtype=np.float32).reshape(7, 9)
        targets = (xyzdmlc[:, :4] * np.float32(10.0)).astype(np.float32)
        _RangeHandler.payloads = {
            "/xyzdmlc.npz": _npz_bytes(case_name, xyzdmlc),
            "/targets.npz": _npz_bytes(case_name, targets),
        }
        _RangeHandler.requests = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}"
        try:
            bundle = MODULE.RemotePointDeepONetBundle(
                targets_url=f"{base_url}/targets.npz",
                xyzdmlc_url=f"{base_url}/xyzdmlc.npz",
            )
            case = MODULE.SelectedCase(
                selection_rank=0,
                case_name=case_name,
                direction="ver",
                item_name="1_1",
                mass_kg=1.0,
                split="train",
                split_rank=0,
            )
            with TemporaryDirectory() as temporary:
                output = Path(temporary)
                input_member = MODULE._resolve_member_offset(
                    bundle.xyzdmlc_remote, bundle.xyzdmlc[case_name]
                )
                target_member = MODULE._resolve_member_offset(
                    bundle.targets_remote, bundle.targets[case_name]
                )
                stage = output / ".staging" / case_name
                stage.mkdir(parents=True)
                input_partial = input_member.size // 2
                target_partial = target_member.size // 2
                (stage / "xyzdmlc.npy.part").write_bytes(
                    _RangeHandler.payloads["/xyzdmlc.npz"][
                        input_member.absolute_data_offset :
                        input_member.absolute_data_offset + input_partial
                    ]
                )
                (stage / "targets.npy.part").write_bytes(
                    _RangeHandler.payloads["/targets.npz"][
                        target_member.absolute_data_offset :
                        target_member.absolute_data_offset + target_partial
                    ]
                )
                _RangeHandler.requests = []
                status = MODULE.prepare_case(
                    output,
                    case,
                    bundle,
                    n_points=5,
                    sampling_seed=42,
                    chunk_size=37,
                )
                self.assertEqual(status, "written")
                self.assertIn(
                    ("/xyzdmlc.npz", input_member.absolute_data_offset + input_partial,
                     input_member.absolute_data_offset + input_partial + 36),
                    _RangeHandler.requests,
                )
                self.assertIn(
                    ("/targets.npz", target_member.absolute_data_offset + target_partial,
                     target_member.absolute_data_offset + target_partial + 36),
                    _RangeHandler.requests,
                )
                case_path = output / "cases" / "train" / f"{case_name}.npz"
                with np.load(case_path, allow_pickle=False) as data:
                    indices = data["sample_indices"]
                    np.testing.assert_array_equal(data["xyzdmlc"], xyzdmlc[indices])
                    np.testing.assert_array_equal(data["targets"], targets[indices])
                    self.assertEqual(len(np.unique(indices)), 5)
                    self.assertEqual(data["xyzdmlc"].dtype, np.float32)
                    self.assertEqual(data["targets"].dtype, np.float32)
                self.assertEqual(
                    MODULE.prepare_case(
                        output,
                        case,
                        bundle,
                        n_points=5,
                        sampling_seed=42,
                        chunk_size=37,
                    ),
                    "skipped",
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
