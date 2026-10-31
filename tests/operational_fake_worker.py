"""Synthetic process faults only. No scenario, model checkpoint or solver access."""
import json
import os
import subprocess
import sys
import time


def main():
    print(json.dumps({"event": "BOOT", "pid": os.getpid()}), flush=True)
    from src.operational_exit.supervisor import enroll_current_process
    enroll_current_process(json.loads(sys.stdin.readline())["job_name"])
    print(json.dumps({"event": "ENROLLED", "pid": os.getpid()}), flush=True)
    settings = json.loads(sys.stdin.readline())
    print(json.dumps({"event": "READY", "metadata": {"synthetic": True}}), flush=True)
    request = json.loads(sys.stdin.readline())
    from src.operational_exit.common import atomic_json, sealed
    mode = settings["mode"]
    payload = dict(identity=request["identity"], compute_end=time.perf_counter(),
                   status="solution", validation={"valid": mode != "invalid"}, dispatch={"synthetic": True})
    if mode == "no_incumbent":
        payload.update(status="no_incumbent", dispatch=None)
    if mode in ("validation_overrun", "write_overrun"):
        time.sleep(2)
    if mode == "crash":
        sys.exit(7)
    if mode == "blocked_child":
        child = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(60)"],
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        payload["child_pid"] = child.pid
        payload["child_executable"] = sys.executable
    atomic_json(request["result_file"], sealed(payload))
    if mode == "written_but_active":
        time.sleep(2)


if __name__ == "__main__":
    main()
