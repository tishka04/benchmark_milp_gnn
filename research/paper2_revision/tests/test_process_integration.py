"""Small CPU integration checks of the actual Windows process supervisor."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from research.paper2_revision.core import ROOT, HOME, sha, source_hashes


@unittest.skipUnless(os.name == "nt", "Production supervisor uses Windows Job Objects")
class ProcessIntegrationTests(unittest.TestCase):
    def test_cpu_workers_validate_and_close_the_process_tree(self):
        sys.path.insert(0, str(ROOT / "tests"))
        from test_milp_load_shifting import scenario_input
        from src.operational_exit.supervisor import run_experiment
        with tempfile.TemporaryDirectory(dir=HOME) as directory:
            folder = Path(directory)
            scenario = folder / "scenario.json"
            scenario.write_text(json.dumps(scenario_input(n=2, zones=1)), encoding="utf-8")
            settings = dict(extension_sources=source_hashes(), model_hashes={}, reserve_s=1.,
                            solver_seconds=2., repair=dict(max_stages=3, slack_tol_mwh=1.,
                            deviation_penalty=10000., flip_budget_20=100, flip_budget_100=1000))
            for method in ("highs", "rh"):
                request = dict(key="synthetic_process_test", budget=5, repetition=0, seed=7,
                               identity=dict(test="process_closure", method=method),
                               file=str(scenario), scenario_sha256=sha(scenario))
                outcome = run_experiment(dict(settings, method=method), request, folder / method,
                    preload_timeout=60, worker_module="research.paper2_revision.worker")
                self.assertTrue(outcome["all_processes_exited"])
                self.assertTrue(outcome["success"], outcome["status"])
                self.assertLessEqual(outcome["exit_s"], request["budget"])
                self.assertTrue(outcome["result"]["validation"]["valid"])


if __name__ == "__main__":
    unittest.main()
