"""Bounded process-level timing regressions; no real inference or solver runs."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from test_milp_load_shifting import scenario_input
from src.operational_exit.common import ROOT, atomic_json, read_json, sha
from src.operational_exit.supervisor import run_experiment


@unittest.skipUnless(os.name == "nt", "Windows Job Object supervision")
class EmbeddingTimingTests(unittest.TestCase):
    def probe(self, method, *, budget, delay, directory):
        scenario = directory / "scenario_synthetic.json"
        trace_path = directory / "embedding_trace.json"
        atomic_json(scenario, scenario_input(n=2, dr_units=0))
        request = dict(budget=budget, key="synthetic/embedding_timing", repetition=0, seed=42,
                       identity={"method": method, "delay": delay}, file=str(scenario), scenario_sha256=sha(scenario))
        settings = dict(method=method, reserve_s=.1, repair={}, embedding_delay_s=delay, trace_file=str(trace_path))
        result = run_experiment(settings, request, directory / "attempt",
                                worker_module="tests.operational_embedding_timing_worker")
        return result, read_json(trace_path)

    def test_embedding_generation_is_inside_compute_and_exit_time(self):
        for method in ("jepa", "pipeline"):
            with self.subTest(method=method), tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
                result, trace = self.probe(method, budget=8., delay=.4, directory=Path(tmp))
                self.assertTrue(result["success"], result)
                self.assertFalse(result["preload_metadata"]["scenario_preloaded"])
                self.assertTrue(result["all_processes_exited"])
                self.assertGreaterEqual(trace["encode_start"], trace["start"])
                self.assertGreaterEqual(trace["encode_end"] - trace["encode_start"], .4)
                self.assertGreaterEqual(result["result"]["compute_end"], trace["encode_end"])
                self.assertGreaterEqual(result["compute_s"], trace["encode_end"] - trace["start"])
                self.assertGreaterEqual(result["exit_s"], result["compute_s"])
                self.assertAlmostEqual(trace["deadline"] - trace["start"], 8.)
                self.assertAlmostEqual(trace["compute_deadline"], trace["deadline"] - .1)
                self.assertEqual(result["result"]["generated_candidates"], 16 if method == "pipeline" else 1)
                print(f"{method}: HTE delay={trace['encode_end']-trace['encode_start']:.3f}s, "
                      f"compute={result['compute_s']:.3f}s, exit={result['exit_s']:.3f}s", flush=True)

    def test_embedding_overrun_is_killed_and_never_counts_as_success(self):
        for method in ("jepa", "pipeline"):
            with self.subTest(method=method), tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
                result, trace = self.probe(method, budget=.6, delay=2., directory=Path(tmp))
                self.assertGreaterEqual(trace["encode_start"], trace["start"])
                self.assertNotIn("encode_end", trace)
                self.assertFalse(result["success"])
                self.assertEqual(result["status"], "forced_timeout")
                self.assertTrue(result["all_processes_exited"])
                self.assertIsNone(result["result"])
                self.assertGreaterEqual(result["exit_s"], .6)
                print(f"{method}: HTE delay=2s > budget=0.6s, {result['status']}, "
                      f"exit={result['exit_s']:.3f}s", flush=True)


if __name__ == "__main__":
    unittest.main()
