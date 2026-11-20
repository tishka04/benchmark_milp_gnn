"""Synthetic regression tests for deterministic, provenance-preserving HTE reuse."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from test_milp_load_shifting import make_data
from src.gnn.hetero_graph_dataset import build_hetero_temporal_record
from src.operational_exit.common import ROOT, atomic_json, digest, read_json, save_record, sha
from src.operational_exit.hte_compatibility import (
    AMENDMENT_KIND, AMENDMENT_NAME, CONTRACT_NAME, INPUT_ARRAYS,
    apply_weather_codes, assert_graph_equal, check_repair_sources,
    load_feature_contract, recover_weather_codes, register_runtime_repair, weather_rows,
)
from src.operational_exit.protocol import effective_sources


class HTECompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.data = make_data(zones=2)
        self.raw = build_hetero_temporal_record(self.data)
        self.old = copy.deepcopy(self.raw)
        self.codes = {p: 17 for p in self.data.region_weather_profile.values()}
        apply_weather_codes(self.old, self.data, self.codes)

    def test_recovered_vocabulary_reproduces_every_historical_input(self):
        recovered = {}
        recover_weather_codes(self.old, self.data, recovered)
        self.assertEqual(recovered, self.codes)
        fixed = apply_weather_codes(copy.deepcopy(self.raw), self.data, recovered)
        assert_graph_equal(self.old, fixed, "synthetic")
        weather = np.asarray(self.raw["node_types"]) == 4
        np.testing.assert_array_equal(fixed["node_features"][~weather], self.raw["node_features"][~weather])
        for key in set(INPUT_ARRAYS) - {"node_features"}:
            np.testing.assert_array_equal(fixed[key], self.raw[key])

    def test_conflicting_codes_are_rejected_instead_of_selected_per_case(self):
        recovered = {}
        recover_weather_codes(self.old, self.data, recovered)
        another = apply_weather_codes(copy.deepcopy(self.raw), self.data, {p: 18 for p in self.codes})
        with self.assertRaisesRegex(ValueError, "Conflicting historical"):
            recover_weather_codes(another, self.data, recovered)

    def test_time_varying_or_fractional_historical_code_is_rejected(self):
        indices, profile = weather_rows(self.old, self.data)[0]
        for value in (18., 17.5, float("nan")):
            broken = copy.deepcopy(self.old)
            broken["node_features"][indices[-1], 1] = value
            with self.assertRaisesRegex(ValueError, "Invalid historical"):
                recover_weather_codes(broken, self.data, {})

    def test_unknown_profile_never_falls_back_to_hash(self):
        with self.assertRaisesRegex(ValueError, "absent from frozen"):
            apply_weather_codes(self.raw, self.data, {})

    def test_other_feature_or_edge_changes_remain_rejected(self):
        for key in ("node_features", "edge_index", "edge_features"):
            with self.subTest(key=key):
                broken = copy.deepcopy(self.old)
                np.asarray(broken[key]).flat[0] += 1
                with self.assertRaisesRegex(ValueError, key):
                    assert_graph_equal(self.old, broken, "synthetic")

    def test_weather_node_identity_is_checked(self):
        broken = copy.deepcopy(self.old)
        indices, profile = weather_rows(broken, self.data)[0]
        broken["node_ids"][indices[-1]] = "Weather_wrong#t=1"
        with self.assertRaisesRegex(ValueError, "axes changed"):
            apply_weather_codes(broken, self.data, self.codes)

    def test_only_unused_line_distance_metadata_may_differ(self):
        old = copy.deepcopy(self.old)
        old["edge_types"][0] = 5
        old["edge_features"] = np.pad(old["edge_features"], ((0, 0), (0, max(0, 2-old["edge_features"].shape[1]))))
        new = copy.deepcopy(old)
        new["edge_features"][0, 1] += 50
        assert_graph_equal(old, new, "synthetic unused distance")
        new["edge_features"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "outside unused"):
            assert_graph_equal(old, new, "synthetic line capacity")

    def test_features_are_identical_across_python_hash_seeds(self):
        script = """
import sys, json, hashlib
sys.path.insert(0, 'tests')
from test_milp_load_shifting import make_data
from src.gnn.hetero_graph_dataset import build_hetero_temporal_record
from src.operational_exit.hte_compatibility import apply_weather_codes
d = make_data(zones=2)
r = build_hetero_temporal_record(d)
raw = hashlib.sha256(r['node_features'].tobytes()).hexdigest()
apply_weather_codes(r, d, {p: 17 for p in d.region_weather_profile.values()})
print(json.dumps({'raw': raw, 'fixed': hashlib.sha256(r['node_features'].tobytes()).hexdigest()}))
"""
        rows = [json.loads(subprocess.check_output([sys.executable, "-B", "-c", script], cwd=ROOT,
                env=dict(os.environ, PYTHONHASHSEED=seed), text=True, timeout=60)) for seed in ("1", "999")]
        self.assertNotEqual(rows[0]["raw"], rows[1]["raw"])
        self.assertEqual(rows[0]["fixed"], rows[1]["fixed"])

    def test_contract_is_bound_to_encoder_and_manifest(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temp:
            folder = Path(temp)
            checkpoint = folder / "encoder.pt"
            checkpoint.write_bytes(b"synthetic encoder identity")
            path = folder / CONTRACT_NAME
            save_record(path, dict(complete=True, kind=AMENDMENT_KIND, hte_sha256=sha(checkpoint),
                                  manifest_hash="m", weather_codes=self.codes))
            self.assertEqual(load_feature_contract(path, checkpoint, "m")["weather_codes"], self.codes)
            with self.assertRaises(ValueError):
                load_feature_contract(path, checkpoint, "other")
            checkpoint.write_bytes(b"different encoder")
            with self.assertRaises(ValueError):
                load_feature_contract(path, checkpoint, "m")

    def test_repair_rejects_any_physics_change(self):
        manifest = {"sources": {"src/milp/model.py": "old", "src/operational_exit/embedding.py": "old"}}
        with self.assertRaises(ValueError):
            check_repair_sources(manifest, {"src/milp/model.py": "new", "src/operational_exit/embedding.py": "new"})

    def test_amendment_retains_original_manifest_and_completed_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temp:
            folder = Path(temp)
            manifest = dict(sources={"src/operational_exit/embedding.py": "before"}, versions={}, train=[], cohort=[])
            atomic_json(folder / "manifest.json", manifest)
            identity = dict(manifest_hash=digest(manifest), key="synthetic")
            save_record(folder / "labels/synthetic.json", dict(identity=identity, complete=True, status="no_incumbent"))
            original_manifest = (folder / "manifest.json").read_bytes()
            original_label = (folder / "labels/synthetic.json").read_bytes()
            atomic_json(folder / "state.json", dict(steps={str(s): "completed" for s in (2, 3, 4)}))
            sources = {"src/operational_exit/embedding.py": "after", "src/operational_exit/hte_compatibility.py": "new"}
            def fake_contract(*args):
                atomic_json(folder / CONTRACT_NAME, {"synthetic": True})
            with patch("src.operational_exit.protocol.source_signature", return_value=sources), \
                 patch("src.operational_exit.protocol.versions", return_value={}), \
                 patch("src.operational_exit.hte_compatibility.ensure_feature_contract", side_effect=fake_contract):
                with self.assertRaises(ValueError):
                    effective_sources(manifest, folder)
                register_runtime_repair(manifest, folder, lambda *a: None)
                self.assertEqual(effective_sources(manifest, folder), sources)
                self.assertEqual((folder / "manifest.json").read_bytes(), original_manifest)
                self.assertEqual((folder / "labels/synthetic.json").read_bytes(), original_label)
                atomic_json(folder / CONTRACT_NAME, {"synthetic": False})
                with self.assertRaises(ValueError):
                    effective_sources(manifest, folder)

    def test_amendment_forbidden_after_downstream_work(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temp:
            folder = Path(temp)
            manifest = dict(sources={"src/operational_exit/embedding.py": "before"}, versions={})
            atomic_json(folder / "state.json", dict(steps={str(s): "completed" for s in (2, 3, 4, 5)}))
            with patch("src.operational_exit.protocol.source_signature", return_value={"src/operational_exit/embedding.py": "after"}), \
                 patch("src.operational_exit.protocol.versions", return_value={}):
                with self.assertRaisesRegex(ValueError, "after embeddings"):
                    register_runtime_repair(manifest, folder, lambda *a: None)


if __name__ == "__main__":
    unittest.main()
