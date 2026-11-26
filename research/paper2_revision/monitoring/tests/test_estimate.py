import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.paper2_revision.core import config, digest, sealed, sha
from research.paper2_revision.monitoring.estimate import (
    Job, Timings, build_report, history_rows, plan, planned_cases, render, terminal_records, training_row)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sealed(value)), encoding="utf-8")


class EstimateTests(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.case = dict(key="development/f/0/base", group="development/f/0", family="f",
                         partition="development", condition="base", expected_horizon=24,
                         scenario_sha256="scenario")

    def trial(self, suite="four", method="pipeline", **changes):
        case = dict(self.case, **changes)
        return Job("Development", "trial", case,
                   dict(key=case["key"], suite=suite, method=method, budget=30, repetition=0), 30)

    def test_schedule_counts_and_inapplicable_cells(self):
        jobs = plan(planned_cases(self.cfg), self.cfg)
        self.assertEqual(sum(j.kind == "reference" for j in jobs.values()), 1200)
        self.assertEqual(sum(j.kind == "trial" for j in jobs.values()), 23700)
        expected = dict(ReferencesDevelopment=330, Development=7200, TrainingLabels=540,
                        ValidateMultiHorizon=3150, ReferencesFinal=330, Final=12150, Gurobi=1200)
        for stage, count in expected.items():
            self.assertEqual(sum(j.stage == stage for j in jobs.values()), count)
        self.assertFalse(any(j.case["expected_horizon"] > 24 and
                             j.item.get("method") in ("jepa", "pipeline", "perturb16") for j in jobs.values()))
        primary, ablation = self.trial(), self.trial(suite="ablation")
        self.assertNotEqual(primary.key, ablation.key)

    def test_timings_include_preload_and_failed_trials(self):
        job = self.trial()
        pools = Timings()
        for _ in range(3):
            pools.add(job.context, (7, 20), historical=True)
        self.assertEqual(pools.forecast(job), (27, "historical"))
        for _ in range(3):
            pools.add(job.context, (10, 30))
        self.assertEqual(pools.forecast(job), (40, "current"))

    def test_new_horizons_methods_and_families_use_provisions(self):
        pools = Timings()
        for _ in range(5):
            pools.add(self.trial().context, (8, 20), historical=True)
        for job in (self.trial(condition="horizon48", expected_horizon=48),
                    self.trial(condition="size"), self.trial(method="mh_pipeline"),
                    self.trial(family="another_family")):
            self.assertEqual(pools.forecast(job), (38, "provision"))
        self.assertEqual(Timings().forecast(self.trial()), (630, "provision_no_preload"))

    def test_small_samples_do_not_give_spurious_precision(self):
        job = self.trial()
        pools = Timings()
        pools.add(job.context, (6, 1))
        self.assertEqual(pools.forecast(job), (36, "provision"))

    def test_count_only_sealed_terminal_matching_records_and_distinct_suites(self):
        snap = {"sha256": "freeze"}
        primary, ablation = self.trial(), self.trial(suite="ablation")
        jobs = {j.key: j for j in (primary, ablation)}
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            identity = dict(primary.item, freeze="freeze", protocol=self.cfg["protocol"],
                            case_sha256="scenario")
            output = dict(identity=identity, complete=True, success=False, status="forced_timeout",
                          preload_s=8., exit_s=30.)
            save(run / "experiments" / (digest(identity) + ".json"), output)
            other = dict(identity, suite="ablation")
            save(run / "experiments" / (digest(other) + ".json"), dict(output, identity=other, complete=False))
            foreign = dict(identity, freeze="another")
            save(run / "experiments" / (digest(foreign) + ".json"), dict(output, identity=foreign))
            (run / "experiments" / "bad.json").write_text("{")
            warnings = []
            done, measured = terminal_records(run, jobs, snap, Timings(), warnings)
            self.assertEqual(set(done), {primary.key})
            self.assertEqual(measured[primary.key], 38)
            self.assertTrue(warnings)

    def test_reference_durations_and_read_only_report(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            snap = sealed(dict(config=self.cfg, manifest_hash="old"))
            save(run / "freeze.json", snap)
            save(run / "cases.json", dict(freeze=snap["sha256"], entries=[self.case]))
            save(run / "references" / (digest(self.case) + ".json"),
                 dict(identity=dict(case=self.case, freeze=snap["sha256"]), complete=True,
                      supervision=dict(preload_s=6., exit_s=12.)))
            before = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
            report = build_report(run, use_history=False)
            after = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(report["completed_reference_files"], 1)
            self.assertEqual(report["measured_seconds"], 18)
            self.assertEqual(report["current_timing_samples"], 1)

    def test_verified_small_historical_table_and_corrupt_table_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            source = run / "original.json"
            original_row = dict(observed=True, family="f", method="highs", budget=30,
                                n_steps=24, preload_s=6, exit_s=20)
            save(source, dict(freeze="old-freeze", rows=[original_row]))
            analysis = run / "analyses" / "historical_test"
            table = analysis / "tables" / "observations.csv"
            table.parent.mkdir(parents=True)
            table.write_text("observed,family,method,budget,n_steps,preload_s,exit_s\nTrue,f,highs,30,24,6,20\n")
            save(analysis / "receipt.json", dict(name="historical",
                source=dict(file=str(source), sha256=sha(source), source_freeze="old-freeze"),
                outputs={"tables/observations.csv": sha(table)}))
            with patch("research.paper2_revision.monitoring.estimate.read", wraps=lambda p: json.loads(Path(p).read_text())) as read_mock:
                self.assertEqual(history_rows(run, source, {"sha256": "old-freeze"}), [original_row])
                self.assertNotIn(source, [call.args[0] for call in read_mock.call_args_list])
            table.write_text("corrupted")
            self.assertEqual(history_rows(run, source, {"sha256": "old-freeze"}), [original_row])

    def test_next_subblock_excludes_later_transformations(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            snap = sealed(dict(config=self.cfg, manifest_hash="old"))
            save(run / "freeze.json", snap)
            entries = [self.case, dict(self.case, key="controlled_development/f/0/base",
                                      partition="controlled_development")]
            save(run / "cases.json", dict(freeze=snap["sha256"], entries=entries))
            save(run / "input_audit.json", {})
            report = build_report(run, use_history=False)
            self.assertEqual(report["next_block"]["partition"], "development")
            self.assertEqual(report["next_block"]["total"], 1)
            self.assertEqual(report["stages"][0]["total"], 2)

    def test_training_receipts_project_only_same_part_and_leave_partial_unknown(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            seeds = self.cfg["training_seeds"]
            model = run / "multi_horizon" / str(seeds[0])
            save(model / "ae_receipt.json", dict(freeze="freeze", wall_seconds=90))
            (model / "ae_best.pt").write_bytes(b"not deserialized")
            second = run / "multi_horizon" / str(seeds[1])
            second.mkdir()
            (second / "ae_last.pt").write_bytes(b"never load pickle to show status")
            row = training_row(run, self.cfg, {"sha256": "freeze"}, [])
            self.assertEqual(row["completed"], 1)
            self.assertEqual(row["estimated_seconds"], 90)
            self.assertEqual(row["unknown_remaining"], 7)

    def test_absent_run_has_complete_plan_without_creating_it(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "absent"
            report = build_report(run, use_history=False, hours_per_day=8)
            self.assertFalse(run.exists())
            self.assertEqual(report["trials"], 23700)
            self.assertEqual(report["next_stage_to_complete"], "Prepare")
            self.assertEqual(report["equivalent_days"], report["remaining_seconds"] / 28800)
            self.assertIn("non estime", render(report))
            self.assertIn("PLUS", render(report))


if __name__ == "__main__":
    unittest.main()
