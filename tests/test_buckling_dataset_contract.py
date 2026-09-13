"""Regression checks for physical input fidelity and withheld-variable exports."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "buckling" / "src"
sys.path.insert(0, str(SRC))
from geometry import ShellMesh
from diag_settle import write_diag
import run_production as production
import pack_hdf5
from validate_dataset import audit
from scoring import crps, energy_score, rank_histogram, self_score, skill


class BucklingDeckTests(unittest.TestCase):
    def test_solver_receives_every_taper_thickness(self):
        for gamma in (0.15, 0.25):
            mesh = ShellMesh(140, 1., thickness_gamma=gamma, elems_per_half_wave=.5)
            with tempfile.TemporaryDirectory() as directory:
                deck = Path(directory) / "taper.inp"
                write_diag(deck, mesh, mesh.coords, 1.5 * mesh.end_shortening_cr)
                text = deck.read_text()
            self.assertIn("*SHELL SECTION, ELSET=EALL, MATERIAL=STEEL, NODAL THICKNESS\n", text)
            rows = text.split("*NODAL THICKNESS\n", 1)[1].split("*BOUNDARY", 1)[0].strip().splitlines()
            self.assertEqual(len(rows), mesh.n_nodes)
            ids, values = zip(*(line.split(",") for line in rows))
            np.testing.assert_array_equal(np.array(ids, dtype=int), np.arange(1, mesh.n_nodes + 1))
            np.testing.assert_allclose(np.array(values, dtype=float), mesh.thickness_at(mesh.z), rtol=1e-13)

    def test_legacy_taper_cannot_be_reused_or_overwritten(self):
        g = dict(shape="cylinder", r_over_t=140, l_over_r=1., alpha_deg=0., thickness_gamma=.25)
        with tempfile.TemporaryDirectory() as directory:
            key = production.geom_key(g) + "_s001"
            path = Path(directory) / key / "draw.npz"
            path.parent.mkdir()
            np.savez(path, disp=np.zeros((2, 3)))
            original = path.read_bytes()
            result = production.one((g, 1, 1.5, 1, directory))
            self.assertFalse(result["ok"])
            self.assertIn("uniform thickness", result["tail"])
            self.assertEqual(path.read_bytes(), original)

    def test_cache_rejects_changed_shortening_and_accepts_matching_provenance(self):
        g = dict(shape="cylinder", r_over_t=140, l_over_r=1., alpha_deg=0., thickness_gamma=.25)
        z = {"run_spec_json": np.array(json.dumps(production.run_spec(g, 1, 1.5)))}
        self.assertIsNone(production.cache_problem(z, g, 1, 1.5))
        self.assertIsNotNone(production.cache_problem(z, g, 1, 1.2))


class BucklingExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.g = dict(shape="cylinder", r_over_t=110, l_over_r=.8,
                      alpha_deg=0., thickness_gamma=0.)
        self.mesh = ShellMesh(**self.g, elems_per_half_wave=production.EPH)

    def build_export(self, count=3):
        """Synthetic terminal states verify storage/invariants, not shell physics."""
        from analyse import radial
        records = []
        for seed in range(1, count + 1):
            mesh = self.mesh
            key = production.geom_key(self.g) + "_s%03d" % seed
            phase = 2 * np.pi * seed / count
            envelope = np.sin(np.pi * mesh.z / mesh.L)
            disp = np.stack([.02 * envelope * np.cos(8 * mesh.theta + phase),
                             .02 * envelope * np.sin(8 * mesh.theta + phase),
                             -1.5 * mesh.end_shortening_cr * mesh.z / mesh.L], axis=1)
            disp[mesh.bottom_nodes] = 0
            disp[mesh.top_nodes, :2] = 0
            w = radial(mesh, disp) / mesh.t
            folder = self.path / key
            folder.mkdir()
            np.savez(folder / "draw.npz", disp=disp.astype(np.float32), w=w.astype(np.float32),
                     imp_C_real=np.ones((3, 3)), imp_C_imag=np.zeros((3, 3)),
                     k1=np.arange(3), k2=np.arange(3), spectrum=np.ones(10),
                     n_dominant=8, n_share=.4, rms_over_t=np.sqrt(np.mean(w ** 2)), drift=.05)
            records.append(dict(self.g, key=key, seed=seed, ok=True, n_dominant=8,
                                n_share=.4, rms_over_t=np.sqrt(np.mean(w ** 2)), drift=.05))
        plan = self.path / "plan.json"
        plan.write_text(json.dumps([dict(self.g, tier="train", n_draws=count, seed0=1, shorten=1.5)]))
        (self.path / "results.json").write_text(json.dumps(records))
        out = self.path / "packed.h5"
        args = ["pack_hdf5.py", "--work-dir", str(self.path), "--plan", str(plan),
                "--tiers", "train", "--out", str(out)]
        with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
            pack_hdf5.main()
        return out

    def check(self, path):
        with contextlib.redirect_stdout(io.StringIO()):
            return audit(path)

    def test_pack_exposes_actual_thickness_and_preserves_static_contract(self):
        path = self.build_export()
        self.assertEqual(self.check(path), ([], []))
        with h5py.File(path) as h:
            for sample in h["data"].values():
                np.testing.assert_array_equal(sample["nodal_data"][9, 0],
                                              self.mesh.thickness_at(self.mesh.z).astype(np.float32))
                self.assertEqual(sample["nodal_data"].shape, (11, 1, self.mesh.n_nodes))

    def test_validator_checks_late_samples_all_input_rows_and_edges(self):
        path = self.build_export(count=61)
        self.assertEqual(self.check(path), ([], []))
        with h5py.File(path, "r+") as h:
            last = h["data"][sorted(h["data"])[-1]]
            last["nodal_data"][0, 0, 10] += .001  # leaked perturbed coordinate
            last["nodal_data"][9, 0, 10] = 1.     # old thickness-ratio contract
            last["mesh_edge"][0, 0] = self.mesh.n_nodes  # beyond old first-40 edge check
            last.attrs["drift"] = np.nan
        failures, _ = self.check(path)
        self.assertTrue(any("D4 inputs" in message for message in failures))
        self.assertTrue(any("D5 graph" in message for message in failures))
        self.assertTrue(any("RMS drift" in message for message in failures))

    def test_validator_rejects_late_shape_and_latent_mismatch(self):
        path = self.build_export()
        with h5py.File(path, "r+") as h:
            sid = sorted(h["data"])[-1]
            sample = h["data"][sid]
            old = sample["nodal_data"][:]
            del sample["nodal_data"]
            sample.create_dataset("nodal_data", data=old[:10])
            del h["latent"][sid]
        failures, _ = self.check(path)
        self.assertTrue(any("nodal_data shape" in message for message in failures))
        self.assertTrue(any("latent sample IDs" in message for message in failures))


class BucklingScoreTests(unittest.TestCase):
    def test_fair_crps_and_energy_estimator_and_unbounded_baseline_skill(self):
        # Pair [-1, 1] and midpoint observation: fair CRPS is 1 - 2/2 = 0;
        # the ordinary empirical-ensemble CRPS is 1 - 2/4 = 1/2.
        self.assertEqual(crps([-1., 1.], 0.), 0.)
        self.assertEqual(crps([-1., 1.], 0., fair=False), .5)
        self.assertEqual(energy_score([[-1.], [1.]], [0.]), 0.)
        # Triangle inequality keeps fair energy scores nonnegative; it does
        # not bound skill by the noisy reference half-split baseline.
        triangle = [[1., 0.], [-.5, np.sqrt(3) / 2], [-.5, -np.sqrt(3) / 2]]
        self.assertGreater(energy_score(triangle, [0., 0.]), 0.)
        self.assertGreater(skill(.9, 1., 2.), 1.)

    def test_tied_forecasts_do_not_force_all_observations_into_first_rank(self):
        h = rank_histogram(np.zeros((3, 1)), np.zeros((10000, 1)), rng=np.random.default_rng(4))
        np.testing.assert_allclose(h, .25, atol=.015)
        self.assertEqual(rank_histogram(np.zeros((3, 1)), [[0]], bins=2).shape, (2,))

    def test_empty_or_nonfinite_scores_are_rejected(self):
        for forecast, observation in (([], 0.), ([1., np.nan], 0.)):
            with self.assertRaises(ValueError):
                crps(forecast, observation)
        with self.assertRaises(ValueError):
            energy_score(np.empty((0, 2)), [0., 0.])
        with self.assertRaises(ValueError):
            self_score(np.ones((4, 2)), splits=0)


if __name__ == "__main__":
    unittest.main()
