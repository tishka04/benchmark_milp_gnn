"""Reserve scaling and outcome-independent migration; no real campaign solves."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.operational_exit.common import (ROOT, BUDGETS, FAMILIES, digest, save_record,
    load_record, atomic_json, sha, sealed)
from src.operational_exit.protocol import schedule
from src.operational_exit.campaign import one_run
from src.operational_exit.budget_reserve import (POLICY, KIND, RECEIPT_NAME, PREVIOUS_RECEIPT,
    effective_reserve, evaluation_schedule, check_sources, register_policy, verify_policy)


class BudgetReserveTests(unittest.TestCase):
    def test_scaling_leaves_compute_at_every_budget_and_keeps_120_seconds(self):
        raw = 38.45964360004291
        expected = [.3204970300003576, .6409940600007152, 1.602485150001788,
                    3.204970300003576, 9.614910900010728, 19.229821800021455, raw]
        for budget, reserve in zip(BUDGETS, expected):
            self.assertAlmostEqual(effective_reserve(budget, raw), reserve)
        self.assertEqual(effective_reserve(120, raw), raw)
        for raw in (0., .05, 30., 38.45964360004291, 240.):
            compute = [b - effective_reserve(b, raw) for b in BUDGETS]
            self.assertEqual(compute, sorted(compute))
            self.assertTrue(all(c >= b / 2 for b, c in zip(BUDGETS, compute)))

    def test_invalid_timing_inputs_are_rejected(self):
        for budget, raw in ((0, 1), (-1, 1), (float("nan"), 1), (1, -1), (1, float("inf"))):
            with self.subTest(budget=budget, raw=raw), self.assertRaises(ValueError):
                effective_reserve(budget, raw)

    def test_worker_solver_and_model_changes_are_rejected(self):
        for name in ("worker.py", "solving.py", "generation.py", "physics.py", "supervisor.py"):
            key = "src/operational_exit/" + name
            with self.subTest(name=name), self.assertRaises(ValueError):
                check_sources({key: "a"}, {key: "b"})

    def fixture(self, out):
        case_file = out / "scenario.fixture"
        case_file.write_text("synthetic scenario identity", encoding="utf-8")
        cases = [dict(key=f"{f}/case", family=f, file=str(case_file), scenario_sha256=sha(case_file)) for f in FAMILIES]
        manifest = dict(cohort=cases, train=[], variability_cases=[c["key"] for c in cases], versions={}, sources={})
        models = {}
        for name in ("hte", "hte_features", "ae", "jepa", "ebm"):
            path = out / (name + ".fixture")
            path.write_text(name, encoding="utf-8")
            models[name] = str(path)
        hashes = {k: sha(v) for k, v in models.items()}
        before = {"src/operational_exit/campaign.py": "old"}
        current = {"src/operational_exit/campaign.py": "new", "src/operational_exit/budget_reserve.py": "new"}
        save_record(out / PREVIOUS_RECEIPT, dict(complete=True, sources=before, models=hashes))
        save_record(out / "calibration.json", dict(complete=True, manifest_hash=digest(manifest),
            reserve_s=38.45964360004291, models=hashes))
        atomic_json(out / "state.json", dict(steps={str(i): "completed" for i in range(1, 10)}, timings=[]))
        records = []
        for item in schedule(manifest):
            if item["key"] != cases[0]["key"] or item["budget"] not in (60, 120) or item["method"] not in ("highs", "pipeline"):
                continue
            identity = dict(item, manifest_hash=digest(manifest))
            success = item["method"] == "highs"
            path = out / "experiments" / (digest(identity) + ".json")
            save_record(path, dict(identity=identity, complete=True, success=success,
                status="success" if success else "forced_timeout", result=None))
            records.append(path)
        return manifest, models, before, current, records

    def test_migration_retains_all_old_results_and_reuses_both_120s_outcomes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="reserve_test_") as tmp:
            out = Path(tmp)
            manifest, models, before, current, originals = self.fixture(out)
            original_hashes = {p: sha(p) for p in originals + [Path(str(p) + ".bak") for p in originals]}
            def effective(m, o, **kw):
                verify_policy(m, o, before, current, **kw)
                return current
            with patch("src.operational_exit.protocol.verify_source_history"), \
                 patch("src.operational_exit.protocol.source_signature", return_value=current), \
                 patch("src.operational_exit.protocol.effective_sources", side_effect=effective), \
                 patch("src.operational_exit.protocol.versions", return_value={}), \
                 patch("src.operational_exit.campaign.model_files", return_value=models):
                receipt = register_policy(manifest, out)
                self.assertEqual(receipt["counts"]["reuse"], 2)
                self.assertEqual(receipt["counts"]["superseded"], 2)
                self.assertEqual(receipt, register_policy(manifest, out))
                calibration = load_record(out / "calibration.json")
                plan = list(evaluation_schedule(manifest, calibration, receipt))
                base = list(schedule(manifest))
                self.assertEqual(len(plan), len(base))
                for old, new in zip(base, plan):
                    self.assertEqual(old == new, old["budget"] == 120)
                selected = [i for i in plan if i["key"] == manifest["cohort"][0]["key"]
                            and i["budget"] in (60, 120) and i["method"] in ("highs", "pipeline")]
                def execute(settings, request, directory):
                    return dict(identity=request["identity"], complete=True, success=False, status="forced_timeout", result=None)
                with patch("src.operational_exit.campaign.settings", return_value={}), \
                     patch("src.operational_exit.supervisor.run_experiment", side_effect=execute) as run:
                    for item in selected:
                        one_run(manifest, out, manifest["cohort"][0], item,
                            item.get("reserve_s", calibration["reserve_s"]), "experiments", calibration["models"])
                    self.assertEqual(run.call_count, 2)  # Only both 60 s trials rerun.
                    for item in selected:
                        one_run(manifest, out, manifest["cohort"][0], item,
                            item.get("reserve_s", calibration["reserve_s"]), "experiments", calibration["models"])
                    self.assertEqual(run.call_count, 2)  # New failures are never rerolled.
                self.assertEqual(original_hashes, {p: sha(p) for p in original_hashes})
                verify_policy(manifest, out, before, current, check_artifacts=True)
                Path(models["ae"]).write_text("tampered", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Models changed"):
                    verify_policy(manifest, out, before, current, check_artifacts=True)

    def test_interrupted_marker_archived_and_fresh_campaign_scaled_without_amendment(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="reserve_test_") as tmp:
            out = Path(tmp)
            manifest, models, before, current, originals = self.fixture(out)
            path = next(p for p in originals if load_record(p)["identity"]["budget"] == 120)
            old = load_record(path)
            old.update(complete=False, status="external_interruption_if_not_completed", success=False)
            save_record(path, old)
            original_hash = sha(path)
            with patch("src.operational_exit.protocol.verify_source_history"), \
                 patch("src.operational_exit.protocol.source_signature", return_value=current), \
                 patch("src.operational_exit.protocol.effective_sources"), \
                 patch("src.operational_exit.protocol.versions", return_value={}), \
                 patch("src.operational_exit.campaign.model_files", return_value=models):
                register_policy(manifest, out)
            archived = out / "archives" / KIND / path.name
            self.assertEqual(sha(archived), original_hash)
            calibration = load_record(out / "calibration.json")
            self.assertEqual(list(evaluation_schedule(manifest, calibration)), list(schedule(manifest)))
            fresh = sealed(dict(calibration, reserve_policy=POLICY))
            revised = list(evaluation_schedule(manifest, fresh))
            self.assertTrue(all("reserve_s" in i for i in revised if i["budget"] < 120))
            self.assertEqual(len(list(evaluation_schedule(manifest, fresh, extra=True))), 6 * 7 * 4 * 2)


if __name__ == "__main__":
    unittest.main()
