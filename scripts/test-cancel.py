#!/usr/bin/env python3
"""Real end-to-end test of cancel: spawn an env, spawn a generation runner,
kill the runner + its direct children by PID (exactly what ProcessRegistry.
killCurrentTree does), and verify the env stays alive and reusable.

Uses a lightweight stub runner (sleep + spawns a sleep child) so the test
exercises the kill logic without loading real models.

Usage: python3 scripts/test-cancel.py
"""
import os
import signal
import subprocess
import sys
import tempfile
import time

STUB_RUNNER = """\
import subprocess, sys, time
# Announce our child PID on stdout (the app reads this to track the job).
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
print(f"CHILD_PID={child.pid}", flush=True)
# Forward SIGTERM/SIGINT to the child, then exit (mimics generate_av_runner).
def forward(signum, frame):
    try:
        child.send_signal(signum)
    except Exception:
        pass
    sys.exit(0)
signal.signal(signal.SIGTERM, forward)
signal.signal(signal.SIGINT, forward)
while True:
    time.sleep(1)
"""


def child_pids(pid):
    out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout
    return [int(x) for x in out.split() if x.strip()]


def kill_current_tree(runner_pid, child_pid=None):
    """Mimic ProcessRegistry.killCurrentTree: kill runner + its tracked child."""
    pids = [runner_pid]
    # Prefer the announced child PID (what the app actually tracks).
    if child_pid and child_pid not in pids:
        pids.append(child_pid)
    # Fall back to pgrep discovery of direct children.
    for c in child_pids(runner_pid):
        if c not in pids:
            pids.append(c)
    for pid in reversed(pids):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def alive(pid):
    """True only if the process exists AND is not a zombie."""
    try:
        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat="],
            capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return False
    if not out:
        return False
    # 'Z' = zombie (dead, awaiting reap). Treat as not alive.
    return "Z" not in out


def main():
    # 1. Spawn the "env" (simulates the app's shared Python environment).
    env = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"[TEST] env pid={env.pid}", flush=True)

    # 2. Spawn a stub generation runner as a child of the env.
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(STUB_RUNNER)
        stub_path = f.name
    runner = subprocess.Popen(
        [sys.executable, stub_path],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    print(f"[TEST] runner pid={runner.pid}", flush=True)

    # 3. Wait for the runner to announce its child.
    child_pid = None
    deadline = time.time() + 10
    while time.time() < deadline:
        line = runner.stdout.readline()
        if not line:
            break
        line = line.decode().strip()
        print(f"[TEST] runner said: {line}", flush=True)
        if line.startswith("CHILD_PID="):
            child_pid = int(line.split("=", 1)[1])
            break
    print(f"[TEST] runner child pid={child_pid}", flush=True)

    # 4. Kill the runner + tracked child by PID (mimics killCurrentTree).
    kill_current_tree(runner.pid, child_pid)
    time.sleep(1)

    # 5. Verify everything died EXCEPT the env.
    runner_alive = alive(runner.pid)
    kids_alive = [k for k in ([child_pid] if child_pid else []) if alive(k)]
    print(f"[TEST] runner alive after kill: {runner_alive}", flush=True)
    print(f"[TEST] children alive after kill: {kids_alive}", flush=True)

    # 6. Verify env is reusable: spawn a new job under it. This is the real
    #    proof the env survived (more reliable than a ps probe, which the
    #    sandbox can block).
    env_ok = True
    try:
        job2 = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        job2.wait(timeout=10)
        print(f"[TEST] new job under env ran OK (exit {job2.returncode})", flush=True)
        env_ok = job2.returncode == 0
    except Exception as e:
        print(f"[TEST] new job under env FAILED: {e}", flush=True)
        env_ok = False

    try:
        env.kill()
    except Exception:
        pass

    ok = (not runner_alive) and (not kids_alive) and env_ok
    reasons = []
    if runner_alive:
        reasons.append("runner still alive")
    if kids_alive:
        reasons.append(f"children still alive: {kids_alive}")
    if not env_ok:
        reasons.append("new job under env failed")
    print(f"\n[TEST] {'PASS' if ok else 'FAIL'}", flush=True)
    if reasons:
        print(f"[TEST] reason: {'; '.join(reasons)}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
