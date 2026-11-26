"""CPU-only checkpoint I/O tests. Never invoke the scientific training loop."""
from contextlib import contextmanager
import errno
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas  # Initialize Arrow before Torch DLLs on Windows.
import numpy as np
import torch

from research.paper2_revision import core, multihorizon
from research.paper2_revision.recovery import training


@contextmanager
def checkpoint_folder():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        with patch.object(core, "HOME", root), patch.object(training, "RETRY_DELAYS", (0, 0)):
            yield root


def saved(path, step):
    torch.save({"steps": step, "model": {"weight": torch.tensor([float(step)])}}, path)


def loaded(path):
    return torch.load(path, map_location="cpu", weights_only=False)


class CheckpointWriterTests(unittest.TestCase):
    def test_transient_primary_lock_retries_publication_without_reserializing(self):
        with checkpoint_folder() as root:
            target = root / "ae_last.pt"
            saved(target, 1)
            original_replace = os.replace
            calls = []

            def replace(source, destination):
                if Path(destination).name == target.name:
                    calls.append((source, destination))
                    if len(calls) < 3:
                        raise PermissionError(errno.EACCES, "temporarily locked")
                return original_replace(source, destination)

            with patch.object(training.os, "replace", side_effect=replace), \
                    patch.object(training.time, "sleep") as sleep, \
                    patch.object(torch, "save", wraps=torch.save) as serialize:
                training.resilient_checkpoint(target, {"steps": 2})
            self.assertEqual(len(calls), 3)
            self.assertEqual(serialize.call_count, 1)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(loaded(target)["steps"], 2)
            self.assertEqual(loaded(target.with_suffix(".pt.bak"))["steps"], 1)

    def test_persistent_primary_lock_retains_original_and_complete_new_temp(self):
        with checkpoint_folder() as root:
            target = root / "ae_last.pt"
            backup = target.with_suffix(".pt.bak")
            saved(target, 1)
            saved(backup, 0)
            before = target.read_bytes()
            original_replace = os.replace

            def replace(source, destination):
                if Path(destination).name == target.name:
                    raise PermissionError(errno.EACCES, "locked throughout retry window")
                return original_replace(source, destination)

            with patch.object(training.os, "replace", side_effect=replace), \
                    patch.object(training.time, "sleep"), self.assertRaises(PermissionError):
                training.resilient_checkpoint(target, {"steps": 2})
            self.assertEqual(target.read_bytes(), before)
            self.assertEqual(loaded(backup)["steps"], 1)
            complete_temps = [p for p in root.glob("*.tmp") if loaded(p).get("steps") == 2]
            self.assertEqual(len(complete_temps), 1)

    def test_partial_backup_copy_never_overwrites_existing_backup(self):
        with checkpoint_folder() as root:
            target = root / "ae_last.pt"
            backup = target.with_suffix(".pt.bak")
            saved(target, 1)
            saved(backup, 0)
            before = {p.name: p.read_bytes() for p in (target, backup)}

            def interrupted_copy(source, destination):
                Path(destination).write_bytes(b"partial copy")
                raise PermissionError(errno.EACCES, "backup source temporarily unavailable")

            with patch.object(training, "_copy_backup", side_effect=interrupted_copy) as copy, \
                    patch.object(training.time, "sleep"), self.assertRaises(PermissionError):
                training.resilient_checkpoint(target, {"steps": 2})
            self.assertEqual(copy.call_count, 3)
            self.assertEqual(before, {p.name: p.read_bytes() for p in (target, backup)})

    def test_backup_publication_failure_preserves_both_committed_files(self):
        with checkpoint_folder() as root:
            target = root / "ae_last.pt"
            backup = target.with_suffix(".pt.bak")
            saved(target, 1)
            saved(backup, 0)
            before = {p.name: p.read_bytes() for p in (target, backup)}
            original_replace = os.replace

            def replace(source, destination):
                if Path(destination).name == backup.name:
                    raise PermissionError(errno.EACCES, "backup locked")
                return original_replace(source, destination)

            with patch.object(training.os, "replace", side_effect=replace), \
                    patch.object(training.time, "sleep"), self.assertRaises(PermissionError):
                training.resilient_checkpoint(target, {"steps": 2})
            self.assertEqual(before, {p.name: p.read_bytes() for p in (target, backup)})

    def test_unrelated_io_error_is_not_retried(self):
        operation = Mock(side_effect=OSError(errno.ENOSPC, "disk full"))
        with patch.object(training.time, "sleep") as sleep, self.assertRaises(OSError):
            training.retry_io(operation, "synthetic failure")
        operation.assert_called_once()
        sleep.assert_not_called()

    def test_windows_sharing_violation_is_retried_even_as_oserror(self):
        error = OSError(errno.EINVAL, "Windows sharing violation")
        error.winerror = 32
        operation = Mock(side_effect=[error, "published"])
        with patch.object(training, "RETRY_DELAYS", (0,)), patch.object(training.time, "sleep"):
            self.assertEqual(training.retry_io(operation, "synthetic share"), "published")
        self.assertEqual(operation.call_count, 2)

    def test_write_outside_research_home_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(core, "HOME", root / "allowed"), self.assertRaises(ValueError):
                training.resilient_checkpoint(root / "outside.pt", {"steps": 1})
            self.assertFalse((root / "outside.pt").exists())

    def test_checkpoint_restores_optimizer_cursor_and_all_cpu_rng_streams(self):
        # Two tiny CPU optimizer steps exercise saved state; no study model or
        # scientific dataset is constructed or trained.
        previous_rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
        try:
            torch.manual_seed(24)
            np.random.seed(24)
            random.seed(24)
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

            def step(network, opt):
                opt.zero_grad(set_to_none=True)
                inputs = torch.randn(4, 3)
                scale = float(np.random.uniform()) + random.random()
                loss = network(inputs).square().mean() * scale
                loss.backward()
                opt.step()

            step(model, optimizer)
            with checkpoint_folder() as root:
                target = root / "ae_last.pt"
                state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    epoch=4, cursor=8, steps=221, identity="synthetic", history=[{"epoch": 3}],
                    torch_rng=torch.get_rng_state(), cuda_rng=[],
                    numpy_rng=np.random.get_state(), python_rng=random.getstate())
                training.resilient_checkpoint(target, state)
                step(model, optimizer)
                expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
                restored = loaded(target)
                resumed = torch.nn.Linear(3, 2)
                resumed.load_state_dict(restored["model"])
                resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01)
                resumed_optimizer.load_state_dict(restored["optimizer"])
                torch.set_rng_state(restored["torch_rng"])
                np.random.set_state(restored["numpy_rng"])
                random.setstate(restored["python_rng"])
                step(resumed, resumed_optimizer)
                for key, value in resumed.state_dict().items():
                    self.assertTrue(torch.equal(value, expected[key]), key)
                self.assertEqual((restored["epoch"], restored["cursor"], restored["steps"]), (4, 8, 221))
        finally:
            torch.set_rng_state(previous_rng[0])
            np.random.set_state(previous_rng[1])
            random.setstate(previous_rng[2])


class TrainingAdapterTests(unittest.TestCase):
    def test_existing_training_loop_receives_writer_and_keeps_seed_stage_order(self):
        with checkpoint_folder() as root:
            original = multihorizon.checkpoint
            calls = []

            def fake_train(run, part, seed):
                self.assertIs(multihorizon.checkpoint, training.resilient_checkpoint)
                calls.append((run, part, seed))

            with patch.object(core, "frozen", return_value={"sha256": "fixed"}), \
                    patch.object(core, "config", return_value={"training_seeds": [1, 2]}), \
                    patch.object(multihorizon, "train", side_effect=fake_train):
                training.run_training(root)
            self.assertIs(multihorizon.checkpoint, original)
            self.assertEqual(calls, [(root, part, seed) for seed in (1, 2) for part in ("ae", "jepa", "ebm")])

    def test_training_error_restores_original_writer(self):
        with checkpoint_folder() as root:
            original = multihorizon.checkpoint
            with patch.object(core, "frozen", return_value={"sha256": "fixed"}), \
                    patch.object(core, "config", return_value={"training_seeds": [1]}), \
                    patch.object(multihorizon, "train", side_effect=RuntimeError("synthetic interruption")), \
                    self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                training.run_training(root)
            self.assertIs(multihorizon.checkpoint, original)

    def test_writer_lock_and_frozen_guard_are_not_bypassed(self):
        with checkpoint_folder() as root:
            with patch.object(core, "lock", side_effect=PermissionError("writer active")), \
                    patch.object(core, "frozen") as frozen, \
                    patch.object(multihorizon, "train") as train, self.assertRaises(PermissionError):
                training.run_training(root)
            frozen.assert_not_called()
            train.assert_not_called()
            with patch.object(core, "frozen", side_effect=ValueError("Source changed")), \
                    patch.object(multihorizon, "train") as train, \
                    self.assertRaisesRegex(ValueError, "Source changed"):
                training.run_training(root)
            train.assert_not_called()

    def test_final_design_lock_blocks_training(self):
        with checkpoint_folder() as root:
            (root / "design_lock.json").write_text("{}", encoding="utf-8")
            with patch.object(core, "frozen", return_value={"sha256": "fixed"}), \
                    patch.object(multihorizon, "train") as train, self.assertRaises(ValueError):
                training.run_training(root)
            train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
