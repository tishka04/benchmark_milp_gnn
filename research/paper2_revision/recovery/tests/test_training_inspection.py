import hashlib
import json
import random
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas  # Initialize Arrow before Torch DLLs on Windows.
import numpy as np
import torch

from research.paper2_revision import core
from research.paper2_revision.recovery.training import inspect_training


class TrainingInspectionTests(unittest.TestCase):
    def fixture(self, run):
        cfg = dict(core.config(), training_seeds=[11])
        folder = run / "multi_horizon" / "11"
        folder.mkdir(parents=True)
        dataset = core.sealed(dict(freeze="fixed", entries=[dict(split="train") for _ in range(9)]))
        (folder.parent / "dataset.json").write_text(json.dumps(dataset))
        training_cfg = dict(cfg["multi_horizon"], seed=11)
        identity = core.digest(dict(freeze="fixed", dataset=dataset["sha256"], stage="ae",
                                    cfg=training_cfg, dependencies={}))
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        state = dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     epoch=0, cursor=8, steps=1, best=float("inf"), best_state=None, history=[],
                     torch_rng=torch.get_rng_state(), cuda_rng=[], numpy_rng=np.random.get_state(),
                     python_rng=random.getstate())
        path = folder / "ae_last.pt"
        torch.save(state, path)
        return cfg, path, state

    def test_inspect_primary_is_read_only_and_ignores_uncommitted_temporary(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            cfg, path, state = self.fixture(run)
            torch.save(dict(state, cursor=9, steps=2), path.with_suffix(".pt.pending.tmp"))
            before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
            with patch.object(core, "frozen", return_value={"sha256": "fixed"}), patch.object(core, "config", return_value=cfg):
                result = inspect_training(run)
            record = result["checkpoints"][0]
            self.assertEqual((record["status"], record["completed_epochs"], record["cursor"], record["optimizer_steps"]),
                             ("resumable", 0, 8, 1))
            self.assertEqual(record["checkpoint_sha256"], hashlib.sha256(before[path]).hexdigest())
            self.assertEqual(before, {p: p.read_bytes() for p in run.rglob("*") if p.is_file()})

    def test_inspection_rejects_foreign_identity_and_inconsistent_progress(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            cfg, path, state = self.fixture(run)
            with patch.object(core, "frozen", return_value={"sha256": "fixed"}), patch.object(core, "config", return_value=cfg):
                torch.save(dict(state, identity="other"), path)
                with self.assertRaisesRegex(ValueError, "identity/structure"):
                    inspect_training(run)
                torch.save(dict(state, steps=99), path)
                with self.assertRaisesRegex(ValueError, "progress"):
                    inspect_training(run)

    def test_report_hash_matches_loaded_bytes_when_publication_advances(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            cfg, path, state = self.fixture(run)
            original = path.read_bytes()
            real_load = torch.load
            def load_then_advance(*args, **kwargs):
                result = real_load(*args, **kwargs)
                torch.save(dict(state, cursor=9, steps=2), path)
                return result
            with patch.object(core, "frozen", return_value={"sha256": "fixed"}), patch.object(core, "config", return_value=cfg), patch.object(torch, "load", side_effect=load_then_advance):
                record = inspect_training(run)["checkpoints"][0]
            self.assertEqual(record["cursor"], 8)
            self.assertEqual(record["checkpoint_sha256"], hashlib.sha256(original).hexdigest())
            self.assertNotEqual(record["checkpoint_sha256"], core.sha(path))


if __name__ == "__main__":
    unittest.main()
