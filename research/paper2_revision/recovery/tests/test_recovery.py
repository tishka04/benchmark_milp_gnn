from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from research.paper2_revision import core, recovery, cases, audit


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


class RecoveryTests(unittest.TestCase):
    def fixture(self, run, *, complete=True):
        cfg = dict(core.config(), new_cases_per_family=1, controlled_cases_per_family=0,
                   training_bases_per_family=0)
        source = run / "case.json"
        save(source, {"range": [0.4, 0.8], "cost": 12})
        entries = [dict(key=f"{partition}/{i}", partition=partition, file=str(source),
                        scenario_sha256=core.sha(source))
                   for partition, n in (("development", 6), ("final", 6), ("gurobi120", 120))
                   for i in range(n)]
        manifest = core.sealed(dict(freeze="fixed", entries=entries))
        save(run / "freeze.json", {"sha256": "fixed"})
        save(run / "planned_counts.json", core.sealed(recovery.planned_blocks()))
        if complete:
            save(run / "cases.json", manifest)
            self.save_audit(run)
        return cfg, manifest

    def save_audit(self, run):
        save(run / "input_audit.json", core.sealed(dict(freeze="fixed", cases_sha256=core.sha(run / "cases.json"))))

    def guards(self, run, cfg):
        stack = ExitStack()
        stack.enter_context(patch.object(core, "HOME", run))
        stack.enter_context(patch.object(core, "config", return_value=cfg))
        stack.enter_context(patch.object(core, "frozen", return_value={"sha256": "fixed"}))
        return stack

    def test_tuple_list_replay_preserves_bytes_and_rejects_real_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            path = run / "case.json"
            with patch.object(core, "HOME", run):
                core.put(path, {"range": (0.4, 0.8), "cost": 12})
                original = path.read_bytes()
                with self.assertRaises(ValueError):
                    core.put(path, {"range": (0.4, 0.8), "cost": 12})
                recovery.publish_case(path, {"range": (0.4, 0.8), "cost": 12})
                self.assertEqual(path.read_bytes(), original)
                with self.assertRaises(ValueError):
                    recovery.publish_case(path, {"range": (0.4, 0.9), "cost": 12})
                self.assertEqual(path.read_bytes(), original)

    def test_complete_preparation_is_verified_without_regeneration(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            cfg, _ = self.fixture(run)
            before = {p.name: p.read_bytes() for p in run.iterdir()}
            with self.guards(run, cfg), patch.object(cases, "make_cases") as generate, patch.object(audit, "input_audit") as audit_inputs:
                recovery.prepare(run)
                generate.assert_not_called()
                audit_inputs.assert_not_called()
                self.assertEqual(recovery.verify_prepared(run)["cases"], 132)
            self.assertEqual(before, {p.name: p.read_bytes() for p in run.iterdir() if p.name != ".writer.lock"})

    def test_partial_preparation_uses_json_adapter_and_restores_writer(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            cfg, manifest = self.fixture(run, complete=False)
            original_put = cases.put
            original_case = (run / "case.json").read_bytes()
            def generate(folder):
                self.assertIs(cases.put, recovery.publish_case)
                cases.put(folder / "case.json", {"range": (0.4, 0.8), "cost": 12})
                cases.put(folder / "cases.json", manifest)
            with self.guards(run, cfg), patch.object(cases, "make_cases", side_effect=generate), patch.object(audit, "input_audit", side_effect=self.save_audit):
                recovery.prepare(run)
                self.assertEqual(recovery.verify_prepared(run)["cases"], 132)
            self.assertIs(cases.put, original_put)
            self.assertEqual((run / "case.json").read_bytes(), original_case)

    def test_changed_case_is_rejected_without_repair(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            cfg, _ = self.fixture(run)
            (run / "case.json").write_text("modified")
            with self.guards(run, cfg), self.assertRaisesRegex(ValueError, "scenario changed"):
                recovery.prepare(run)
            self.assertEqual((run / "case.json").read_text(), "modified")

    def test_foreign_audit_and_wrong_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            cfg, manifest = self.fixture(run)
            save(run / "input_audit.json", core.sealed(dict(freeze="other", cases_sha256=core.sha(run / "cases.json"))))
            with self.guards(run, cfg), self.assertRaisesRegex(ValueError, "Input audit"):
                recovery.prepare(run)
            manifest["entries"] = manifest["entries"][:-1]
            save(run / "cases.json", core.sealed(manifest))
            with self.guards(run, cfg), self.assertRaisesRegex(ValueError, "cohort counts"):
                recovery.prepare(run)

    def test_frozen_guard_is_not_bypassed(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            cfg, _ = self.fixture(run)
            with self.guards(run, cfg), patch.object(core, "frozen", side_effect=ValueError("Source changed")), self.assertRaisesRegex(ValueError, "Source changed"):
                recovery.prepare(run)

    def test_read_only_verification_does_not_create_a_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            cfg, _ = self.fixture(run)
            with self.guards(run, cfg):
                result = recovery.verify_prepared(run)
            self.assertEqual(result["status"], "verified")
            self.assertFalse((run / ".writer.lock").exists())

    def test_powershell_resume_routes_preparation_and_stops_on_error(self):
        shell = shutil.which("powershell") or shutil.which("pwsh")
        if not shell:
            self.skipTest("PowerShell is not installed")
        launcher = Path(__file__).resolve().parents[2] / "run_paper2.ps1"
        for locked, fail in ((False, False), (True, False), (False, True)):
            with self.subTest(locked=locked, fail=fail), tempfile.TemporaryDirectory() as folder:
                run = Path(folder)
                stub = run / "fake_python.ps1"
                log = run / "calls.jsonl"
                stub.write_text(
                    "Add-Content -LiteralPath $env:P2_TEST_LOG -Encoding utf8 -Value "
                    "(@{ args = @($args) } | ConvertTo-Json -Compress)\n"
                    "$global:LASTEXITCODE = 0\n"
                    "if ($env:P2_FAIL_PREPARE -eq '1' -and $args -contains 'research.paper2_revision.recovery') "
                    "{ $global:LASTEXITCODE = 17 }\n", encoding="utf-8")
                if locked:
                    (run / "design_lock.json").write_text("{}")
                quote = lambda value: "'" + str(value).replace("'", "''") + "'"
                command = f"& {quote(launcher)} -Stage Resume -Run {quote(run)} -Python {quote(stub)}"
                proc = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-Command", command],
                    env=dict(os.environ, P2_TEST_LOG=str(log), P2_FAIL_PREPARE="1" if fail else "0"),
                    capture_output=True, timeout=30)
                calls = [json.loads(line)["args"] for line in log.read_text(encoding="utf-8-sig").splitlines()]
                stages = ["Prepare" if "--stage" not in args else args[args.index("--stage") + 1]
                          for args in calls]
                if fail:
                    self.assertNotEqual(proc.returncode, 0)
                    self.assertEqual(stages, ["Check", "Prepare"])
                else:
                    self.assertEqual(proc.returncode, 0, proc.stderr.decode(errors="replace"))
                    expected = ["Check"]
                    if not locked:
                        expected += ["Prepare", "ReferencesDevelopment", "Development", "TrainingLabels",
                                     "TrainMultiHorizon", "ValidateMultiHorizon", "Lock"]
                    expected += ["ReferencesFinal", "Final", "Gurobi", "Analyze", "Manuscript"]
                    self.assertEqual(stages, expected)
                    if not locked:
                        training_calls = [args for args in calls if "TrainMultiHorizon" in args]
                        self.assertEqual(len(training_calls), 1)
                        self.assertIn("research.paper2_revision.recovery", training_calls[0])


if __name__ == "__main__":
    unittest.main()
