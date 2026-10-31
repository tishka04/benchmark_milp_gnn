"""Measure process-tree EXIT, with a Windows Job Object as the ownership boundary.

The worker cannot preload or spawn children until it belongs to the job. READY
is returned only after preload. The timed request is the first disclosure of
the scenario. A written result does not stop the clock. No late recovery.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid

from .common import ROOT, THREAD_ENV, read_json, verify


class WindowsJob:
    def __init__(self, process):
        if os.name != "nt":
            raise RuntimeError("This campaign requires Windows Job Object supervision")
        k = self.k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k.CreateJobObjectW.restype = wintypes.HANDLE
        for name, args in (
            ("SetInformationJobObject", [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]),
            ("QueryInformationJobObject", [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]),
            ("AssignProcessToJobObject", [wintypes.HANDLE, wintypes.HANDLE]),
            ("TerminateJobObject", [wintypes.HANDLE, wintypes.UINT]),
            ("CloseHandle", [wintypes.HANDLE]),
        ):
            getattr(k, name).argtypes = args
            getattr(k, name).restype = wintypes.BOOL
        class BasicLimits(ctypes.Structure):
            _fields_ = [("user", ctypes.c_int64), ("job_user", ctypes.c_int64),
                        ("flags", wintypes.DWORD), ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t),
                        ("active_limit", wintypes.DWORD), ("affinity", ctypes.c_size_t),
                        ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]
        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io", ctypes.c_uint64 * 6),
                        ("process_mem", ctypes.c_size_t), ("job_mem", ctypes.c_size_t),
                        ("peak_process_mem", ctypes.c_size_t), ("peak_job_mem", ctypes.c_size_t)]
        self.name = "Local\\operational-exit-" + uuid.uuid4().hex
        self.handle = k.CreateJobObjectW(None, self.name)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())
        # Keep the job empty until the actual interpreter self-enrols. A Windows
        # venv redirector may create its own nested job; enrolling the redirector
        # here would make the already-started interpreter's jobs incompatible.
        # The redirector is tracked separately through its retained Popen handle.

    def active(self):
        class Accounting(ctypes.Structure):
            _fields_ = [("times", ctypes.c_int64 * 4), ("faults", wintypes.DWORD),
                        ("total", wintypes.DWORD), ("active", wintypes.DWORD), ("terminated", wintypes.DWORD)]
        info = Accounting()
        if not self.k.QueryInformationJobObject(self.handle, 1, ctypes.byref(info), ctypes.sizeof(info), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(info.active)

    def terminate(self):
        if not self.k.TerminateJobObject(self.handle, 124):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.k.CloseHandle(self.handle)
            self.handle = None


def enroll_current_process(name):
    """Self-enrol the actual interpreter, including behind a venv redirector.

    This needs no cross-process OpenProcess privileges. The parent retains the
    sole long-lived job handle. Children are born inside this job thereafter.
    """
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenJobObjectW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    k.OpenJobObjectW.restype = wintypes.HANDLE
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = k.OpenJobObjectW(0x0001 | 0x0004, False, name)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        already = wintypes.BOOL()
        if not k.IsProcessInJob(k.GetCurrentProcess(), handle, ctypes.byref(already)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not already.value and not k.AssignProcessToJobObject(handle, k.GetCurrentProcess()):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        k.CloseHandle(handle)


def classify(payload, *, budget, exit_s, forced, returncode):
    """Pure acceptance rule, also tested independently of wall-clock scheduling."""
    if forced:
        return False, "forced_timeout"
    if exit_s > budget:
        return False, "late_exit"
    if returncode != 0:
        return False, "worker_error"
    if not payload:
        return False, "missing_final_result"
    if payload.get("status") != "solution":
        return False, payload.get("status", "no_incumbent")
    if not payload.get("validation", {}).get("valid"):
        return False, "invalid_dispatch"
    if not payload.get("dispatch"):
        return False, "missing_dispatch"
    return True, "success"


def worker_command(module):
    """Launch the actual interpreter with the current venv's import paths.

    Windows' venv redirector owns another kill-on-close job and may silently
    kill detached grandchildren when its interpreter returns. Avoid that hidden
    termination path: this supervisor must be the sole lifecycle authority.
    Multiprocessing inherits these exact import paths in its spawn preparation.
    """
    if os.name == "nt" and sys.prefix != sys.base_prefix:
        bootstrap = (
            "import json,runpy,sys; "
            "sys.path[:]=json.loads(sys.argv[1]); "
            "sys.prefix=sys.exec_prefix=sys.argv[2]; "
            "module=sys.argv[3]; sys.argv=[module]; "
            "runpy.run_module(module,run_name='__main__',alter_sys=True)"
        )
        paths = [str(ROOT) if not p else p for p in sys.path]
        return [sys._base_executable, "-B", "-u", "-c", bootstrap, json.dumps(paths), sys.prefix, module]
    return [sys.executable, "-B", "-u", "-m", module]


def run_experiment(settings, request, directory, *, preload_timeout=300, worker_module="src.operational_exit.worker"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    result_file = directory / "final.json"
    preload_start = time.perf_counter()
    env = dict(os.environ, **{k: "1" for k in THREAD_ENV}, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    log = (directory / "worker.log").open("w", encoding="utf-8")
    process = subprocess.Popen(worker_command(worker_module), cwd=ROOT,
                               env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                               text=True, encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    job = None
    events = queue.Queue()
    def reader():
        for line in process.stdout:
            try:
                payload = json.loads(line)
                if isinstance(payload, dict) and payload.get("event") in {"BOOT", "ENROLLED", "READY", "ERROR"}:
                    events.put(payload)
                else:
                    log.write(line)
            except ValueError:
                log.write(line)
        events.put({"error": "Worker closed control stream"})
    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    def receive(kind):
        timeout = max(.001, preload_start + preload_timeout - time.perf_counter())
        item = events.get(timeout=timeout)
        if item.get("event") != kind:
            raise RuntimeError(f"Preload failure: {item}; see {directory / 'worker.log'}")
        return item
    def send(obj):
        process.stdin.write(json.dumps(obj, allow_nan=False) + "\n")
        process.stdin.flush()
    started = None
    try:
        receive("BOOT")
        job = WindowsJob(process)
        send({"job_name": job.name})
        receive("ENROLLED")
        # settings must contain model/runtime configuration only, never a scenario.
        forbidden = {"scenario_file", "case", "label_file", "embedding_file", "scenario_id"}
        if forbidden & settings.keys():
            raise ValueError("Scenario-dependent preload configuration")
        send(settings)
        ready = receive("READY")
        preload_s = time.perf_counter() - preload_start
        budget = float(request["budget"])
        started = time.perf_counter()
        deadline = started + budget
        send(dict(request, start=started, deadline=deadline, result_file=str(result_file)))
        forced = False
        while True:
            # Conservative observation: only a fully empty job can end the clock.
            active = job.active()
            now = time.perf_counter()
            if active == 0 and process.poll() is not None:
                exit_s = now - started
                break
            if now >= deadline:
                forced = True
                job.terminate()
                if process.poll() is None:
                    process.kill()
                break
            time.sleep(min(.001, deadline - now))
        if forced:
            cleanup_start = time.perf_counter()
            while job.active() or process.poll() is None:
                if time.perf_counter() - cleanup_start > 30:
                    raise RuntimeError("Cannot confirm termination of the whole Job Object")
                time.sleep(.005)
            exit_s = time.perf_counter() - started
        process.wait(timeout=5)
        payload = None
        # Never open a timed-out result, even if it was written before a forced kill.
        if not forced and exit_s <= budget and result_file.exists():
            try:
                payload = verify(read_json(result_file))
                if payload.get("identity") != request["identity"]:
                    payload = None
            except (ValueError, UnicodeError):
                payload = None
        success, reason = classify(payload, budget=budget, exit_s=exit_s, forced=forced, returncode=process.returncode)
        compute_end = payload.get("compute_end") if payload else None
        return dict(complete=True, success=success, status=reason, identity=request["identity"],
                    preload_s=preload_s, exit_s=exit_s, deadline_overshoot_s=max(0, exit_s - budget),
                    compute_s=None if compute_end is None else max(0, compute_end - started),
                    finalization_s=None if compute_end is None else max(0, started + exit_s - compute_end),
                    forced=forced, all_processes_exited=job.active() == 0 and process.poll() is not None,
                    returncode=process.returncode, result=payload, preload_metadata=ready.get("metadata", {}))
    finally:
        if job is not None:
            if job.active():
                job.terminate()
            job.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=30)
        process.stdin.close()
        thread.join(timeout=5)
        process.stdout.close()
        log.close()
