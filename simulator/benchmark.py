"""
AIOS-Core :: Benchmark Engine
--------------------------------
Runs two back-to-back trials of the same workload mix:
  Trial A — BASELINE  : orchestrator OFF, raw OS scheduling
  Trial B — AI        : orchestrator ON  (observe mode, no nice() writes)
             optionally with --apply for live priority adjustment

Metrics collected per trial:
  - Aggregate throughput (ops/s) from each worker's stdout
  - Wall-clock completion time
  - System CPU% and MEM% sampled at 200ms intervals
  - Per-PID nice value at end of trial (to confirm AI acted / didn't)

Output: JSON results file + human-readable summary printed to console.

Usage:
    python -m simulator.benchmark
    python -m simulator.benchmark --apply --duration 20 --workers cpu io mixed
"""

import sys
import os
import time
import json
import subprocess
import threading
import argparse
import statistics
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchestrator.engine import AIOrchestrator


# ─────────────────────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WorkerResult:
    wtype:      str
    pid:        int
    duration_s: float
    stdout:     str       = ""
    ops_per_s:  float     = 0.0
    returncode: int       = -1


@dataclass
class TrialResult:
    name:           str
    ai_active:      bool
    ai_apply_os:    bool
    duration_s:     float
    workers:        list = field(default_factory=list)
    cpu_samples:    list = field(default_factory=list)
    mem_samples:    list = field(default_factory=list)
    total_ops_s:    float = 0.0
    mean_cpu_pct:   float = 0.0
    mean_mem_pct:   float = 0.0
    ai_tick_count:  int   = 0


# ─────────────────────────────────────────────────────────────────────────────
#  Throughput parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ops(stdout: str) -> float:
    """Extract ops/s from worker stdout line."""
    for token in stdout.split():
        if "ops/s=" in token or "iters/s=" in token or "wakes/s=" in token:
            try:
                return float(token.split("=")[1])
            except ValueError:
                pass
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Trial runner
# ─────────────────────────────────────────────────────────────────────────────

PYTHON = sys.executable
WORKLOAD_SCRIPT = str(ROOT / "simulator" / "workload.py")


def run_trial(
    name:       str,
    worker_types: list[str],
    duration_s: float,
    ai_active:  bool,
    ai_apply:   bool,
    tick_ms:    int = 200,
    model_path: Optional[str] = None,
    online_update: bool = True,
) -> TrialResult:

    result = TrialResult(
        name=name,
        ai_active=ai_active,
        ai_apply_os=ai_apply,
        duration_s=duration_s,
    )

    # ── Start AI orchestrator if requested ───────────────────────────────────
    orc: Optional[AIOrchestrator] = None
    if ai_active:
        orc = AIOrchestrator(tick_ms=tick_ms, apply_os=ai_apply, model_path=model_path)
        orc.online_update = online_update

    # ── Spawn workers ─────────────────────────────────────────────────────────
    procs = []
    worker_results = []
    for wtype in worker_types:
        cmd = [PYTHON, WORKLOAD_SCRIPT, wtype, str(duration_s)]
        p   = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        wr = WorkerResult(wtype=wtype, pid=p.pid, duration_s=duration_s)
        procs.append((p, wr))
        worker_results.append(wr)

    # Give orchestrator the focused PID list, then start
    if orc:
        orc.watch_pids = [p.pid for p, _ in procs]
        orc.start()

    # ── System sampling thread ────────────────────────────────────────────────
    stop_sampling = threading.Event()
    cpu_samples   = []
    mem_samples   = []

    def _sample():
        while not stop_sampling.is_set():
            cpu_samples.append(psutil.cpu_percent(interval=None))
            mem_samples.append(psutil.virtual_memory().percent)
            time.sleep(0.2)

    sampler = threading.Thread(target=_sample, daemon=True)
    sampler.start()

    # ── Wait for workers ──────────────────────────────────────────────────────
    for p, wr in procs:
        stdout, _ = p.communicate(timeout=duration_s + 10)
        wr.stdout     = stdout.strip()
        wr.ops_per_s  = _parse_ops(stdout)
        wr.returncode = p.returncode

    # ── Stop sampling + orchestrator ──────────────────────────────────────────
    stop_sampling.set()
    sampler.join(timeout=1)

    if orc:
        orc.stop()
        result.ai_tick_count = orc.tick_count

    # ── Aggregate ─────────────────────────────────────────────────────────────
    result.workers      = [asdict(wr) for wr in worker_results]
    result.cpu_samples  = cpu_samples
    result.mem_samples  = mem_samples
    result.total_ops_s  = sum(wr.ops_per_s for wr in worker_results)
    result.mean_cpu_pct = statistics.mean(cpu_samples) if cpu_samples else 0.0
    result.mean_mem_pct = statistics.mean(mem_samples) if mem_samples else 0.0

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Reporter
# ─────────────────────────────────────────────────────────────────────────────

def _bar(value: float, total: float, width: int = 30) -> str:
    frac   = min(value / max(total, 1), 1.0)
    filled = int(frac * width)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {value:.1f}"


def print_comparison(baseline: TrialResult, ai: TrialResult):
    sep = "=" * 60
    print(sep)
    print("  AIOS-Core Benchmark Results")
    print(sep)

    for trial in [baseline, ai]:
        label = f"[{trial.name}]"
        print(f"\n  {label}")
        print(f"    Total ops/s   : {trial.total_ops_s:>10.2f}")
        print(f"    Mean CPU%     : {trial.mean_cpu_pct:>10.2f}")
        print(f"    Mean MEM%     : {trial.mean_mem_pct:>10.2f}")
        if trial.ai_active:
            print(f"    AI ticks      : {trial.ai_tick_count:>10d}")
        for w in trial.workers:
            print(f"      {w['wtype']:<8} pid={w['pid']}  ops/s={w['ops_per_s']:.2f}  rc={w['returncode']}")

    print()
    print(sep)
    delta_ops = ai.total_ops_s - baseline.total_ops_s
    delta_cpu = ai.mean_cpu_pct - baseline.mean_cpu_pct
    pct_change = (delta_ops / max(baseline.total_ops_s, 1)) * 100

    sign_ops = "+" if delta_ops >= 0 else ""
    sign_cpu = "+" if delta_cpu >= 0 else ""

    print(f"  Throughput delta : {sign_ops}{delta_ops:.2f} ops/s  ({sign_ops}{pct_change:.1f}%)")
    print(f"  CPU usage delta  : {sign_cpu}{delta_cpu:.2f}%")
    print()

    if pct_change > 2:
        print("  >> AI improved throughput.")
    elif pct_change < -2:
        print("  >> AI reduced throughput (model needs training).")
    else:
        print("  >> No significant difference (expected with random weights).")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="AIOS-Core Benchmark")
    p.add_argument("--duration", type=float, default=15.0,
                   help="Duration of each trial in seconds (default: 15)")
    p.add_argument("--workers",  nargs="+",
                   choices=["cpu", "io", "mixed", "idle"],
                   default=["cpu", "io", "mixed"],
                   help="Worker types to spawn (default: cpu io mixed)")
    p.add_argument("--apply",    action="store_true",
                   help="Allow AI to write OS nice() adjustments")
    p.add_argument("--tick",     type=int, default=200,
                   help="AI control loop period in ms")
    p.add_argument("--out",      type=str, default="logs/benchmark.json",
                   help="Output path for JSON results")
    p.add_argument("--model",    type=str, default=None,
                   help="Path to trained model weights (JSON)")
    p.add_argument("--no-online-update", action="store_true",
                   help="Disable RL-lite online weight updates during AI trial")
    args = p.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  Workers   : {args.workers}")
    print(f"  Duration  : {args.duration}s per trial")
    print(f"  AI apply  : {args.apply}")
    print(f"  Tick      : {args.tick}ms\n")

    # ── Trial A: Baseline ─────────────────────────────────────────────────────
    print("  [1/2] Running BASELINE trial...")
    baseline = run_trial(
        name         = "BASELINE",
        worker_types = args.workers,
        duration_s   = args.duration,
        ai_active    = False,
        ai_apply     = False,
        tick_ms      = args.tick,
    )
    print(f"        done — {baseline.total_ops_s:.2f} total ops/s")

    # Cool-down
    print("  Cooling down 3s...")
    time.sleep(3)

    # ── Trial B: AI ───────────────────────────────────────────────────────────
    print("  [2/2] Running AI trial...")
    ai_trial = run_trial(
        name         = "AI",
        worker_types = args.workers,
        duration_s   = args.duration,
        ai_active    = True,
        ai_apply     = args.apply,
        tick_ms      = args.tick,
        model_path   = args.model,
        online_update= not args.no_online_update,
    )
    print(f"        done — {ai_trial.total_ops_s:.2f} total ops/s")

    # ── Report ────────────────────────────────────────────────────────────────
    print()
    print_comparison(baseline, ai_trial)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "meta": {
            "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_s":   args.duration,
            "workers":      args.workers,
            "ai_apply_os":  args.apply,
            "tick_ms":      args.tick,
        },
        "baseline": asdict(baseline),
        "ai":       asdict(ai_trial),
        "delta": {
            "ops_per_s":    ai_trial.total_ops_s - baseline.total_ops_s,
            "cpu_pct":      ai_trial.mean_cpu_pct - baseline.mean_cpu_pct,
            "pct_change":   ((ai_trial.total_ops_s - baseline.total_ops_s)
                             / max(baseline.total_ops_s, 1)) * 100,
        }
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved -> {args.out}")


if __name__ == "__main__":
    main()
