"""Run a command and kill it if macOS memory pressure turns red (critical); log what memory did.

Usage: python scripts/mem_guard.py --log <file.json> [--min-free 5] -- <command ...>
Waits up to 5 min for normal pressure before starting, then polls every 0.5 s: kern.memorystatus_vm_pressure_level (1 normal / green, 2 warning / yellow, 4 critical / red) and
memory_pressure's "free percentage". Kills the command (SIGTERM, then SIGKILL) on level 4 or when free memory falls below
--min-free percent. Writes the worst level, lowest free %, swap used (start / max) and the outcome to --log.
Exit code: the command's, or 99 if it was aborted.
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import time


def level() -> int:
    return int(subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True, text=True).stdout)


def free_pct() -> float:
    out = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage:\s*(\d+)", out)
    return float(m.group(1)) if m else 100.0


def swap_mb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) if m else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--min-free", type=float, default=5.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    for _ in range(600):                      # wait up to 5 min for green before starting
        if level() == 1:
            break
        time.sleep(0.5)
    rec = {"cmd": cmd, "level_start": level(), "free_start": free_pct(), "swap_start_mb": swap_mb(),
           "worst_level": 0, "min_free": 100.0, "max_swap_mb": 0.0, "aborted": False, "reason": None}
    if rec["level_start"] >= 2:
        rec.update(aborted=True, reason=f"not started: pressure level {rec['level_start']} before the run")
        json.dump(rec, open(args.log, "w"), indent=2)
        print(f"[mem_guard] {rec['reason']}")
        sys.exit(99)
    t0 = time.time()
    proc = subprocess.Popen(cmd)
    while proc.poll() is None:
        lv, fr, sw = level(), free_pct(), swap_mb()
        rec["worst_level"], rec["min_free"] = max(rec["worst_level"], lv), min(rec["min_free"], fr)
        rec["max_swap_mb"] = max(rec["max_swap_mb"], sw)
        if lv >= 4 or fr < args.min_free:
            rec.update(aborted=True, reason=f"pressure level {lv}, free {fr:.0f}% after {time.time() - t0:.0f} s")
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
        time.sleep(0.5)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["returncode"] = proc.wait()
    json.dump(rec, open(args.log, "w"), indent=2)
    print(f"[mem_guard] worst level {rec['worst_level']}, min free {rec['min_free']:.0f}%, swap "
          f"{rec['swap_start_mb']:.0f} -> max {rec['max_swap_mb']:.0f} MB" + (f"; ABORTED: {rec['reason']}" if rec["aborted"] else ""))
    sys.exit(99 if rec["aborted"] else rec["returncode"])


if __name__ == "__main__":
    main()
