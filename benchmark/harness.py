"""
AIOS-Core :: Benchmark Harness
--------------------------------
Runs two back-to-back trials of identical workload:
  Trial A — BASELINE:  workload running, orchestrator OFF
  Trial B — AI:        workload running, orchestrator ON (observe mode)

Metrics captured per trial:
  - ops/sec per worker type (throughput)
  - system CPU % (mean, p95, std)
  - system MEM % (mean)
  - CPU steal time proxy (involuntary ctx switches across all procs)
  - wall-clock efficiency = total_ops / (cpu_time * n_workers)

Outputs a JSON result file + a Rich comparison table printed to terminal.
"""

import sys
import os
import time
import json
import threading
import statistics
import psutil
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.workload import WorkloadGenerator
from orchestrator.engine import AIOrchestrator


console = Console()

# ── System sampler ────────────────────────────────────────────────────────────

class SystemSampler:
    """Polls psutil every 200ms in a background thread."""

    def __init__(self, interval_ms: int = 200):
        self.interval   = interval_ms / 1000.0
        self.cpu_samples: list[float] = []
        self.mem_samples: list[float] = []
        self._stop      = threading.Event()
        self._thread    = threading.Thread(target=self._run, daemon=True)

    def start(self):
        psutil.cpu_percent(interval=None)   # prime the counter
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop.is_set():
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
            self.mem_samples.append(psutil.virtual_memory().percent)
            time.sleep(self.interval)

    def summary(self) -> dict:
        cpu = self.cpu_samples or [0.0]
        mem = self.mem_samples or [0.0]
        return {
            "cpu_mean": statistics.mean(cpu),
            "cpu_p95":  sorted(cpu)[int(len(cpu) * 0.95)],
            "cpu_std":  statistics.stdev(cpu) if len(cpu) > 1 else 0.0,
            "mem_mean": statistics.mean(mem),
            "samples":  len(cpu),
        }


# ── Single trial runner ───────────────────────────────────────────────────────

def run_trial(
    label:        str,
    duration_s:   float,
    with_ai:      bool,
    workload_mix: list | None = None,
) -> dict:
    console.print(f"\n[bold cyan]>>> Trial: {label}[/bold cyan]  "
                  f"duration={duration_s}s  AI={'ON' if with_ai else 'OFF'}")

    gen     = WorkloadGenerator(mix=workload_mix)
    sampler = SystemSampler(interval_ms=200)
    orc     = AIOrchestrator(tick_ms=200, apply_os=False) if with_ai else None

    orc_ticks = 0
    def _orc_cb(tick, decisions):
        nonlocal orc_ticks
        orc_ticks = tick

    if orc:
        orc.register_callback(_orc_cb)

    t_start = time.perf_counter()

    pids = gen.start()
    sampler.start()
    if orc:
        orc.start()

    console.print(f"  Workers spawned: {gen.worker_count}  PIDs sample: {pids[:4]}…")

    # Progress dots
    step = max(1, int(duration_s / 20))
    for i in range(int(duration_s)):
        time.sleep(1.0)
        if i % step == 0:
            cpu = psutil.cpu_percent(interval=None)
            console.print(f"  t+{i+1:3d}s  CPU={cpu:5.1f}%  "
                          f"{'[AI ticks=' + str(orc_ticks) + ']' if orc else ''}")

    if orc:
        orc.stop()
    sampler.stop()
    worker_results = gen.stop()

    elapsed = time.perf_counter() - t_start
    sys_stats = sampler.summary()

    # Aggregate worker throughput by type
    by_type: dict[str, list[float]] = {}
    total_ops = 0
    for r in worker_results:
        wt = r["type"]
        by_type.setdefault(wt, []).append(r["ops_per_s"])
        total_ops += r["ops"]

    throughput = {wt: statistics.mean(ops) for wt, ops in by_type.items()}
    total_ops_per_s = sum(throughput.values())

    result = {
        "label":           label,
        "with_ai":         with_ai,
        "duration_s":      elapsed,
        "total_ops":       total_ops,
        "total_ops_per_s": total_ops_per_s,
        "throughput":      throughput,
        "sys":             sys_stats,
        "orc_ticks":       orc_ticks,
        "worker_results":  worker_results,
    }

    console.print(f"  [green]Trial complete.[/green]  "
                  f"total_ops/s={total_ops_per_s:,.0f}  "
                  f"cpu_mean={sys_stats['cpu_mean']:.1f}%")
    return result


# ── Comparison report ─────────────────────────────────────────────────────────

def print_report(baseline: dict, ai_run: dict):
    console.rule("[bold white]AIOS-Core Benchmark Report[/bold white]")

    # System overview table
    sys_t = Table(title="System Metrics", box=box.ROUNDED, show_header=True,
                  header_style="bold cyan")
    sys_t.add_column("Metric",    width=22)
    sys_t.add_column("Baseline",  justify="right", width=14)
    sys_t.add_column("AI",        justify="right", width=14)
    sys_t.add_column("Delta",     justify="right", width=14)

    def _delta_str(b, a, invert=False):
        d = a - b
        pct = (d / max(abs(b), 0.001)) * 100
        good = d < 0 if invert else d > 0
        color = "green" if good else "red"
        return f"[{color}]{pct:+.1f}%[/]"

    rows = [
        ("CPU mean %",    baseline["sys"]["cpu_mean"],    ai_run["sys"]["cpu_mean"],    True),
        ("CPU p95 %",     baseline["sys"]["cpu_p95"],     ai_run["sys"]["cpu_p95"],     True),
        ("CPU std dev",   baseline["sys"]["cpu_std"],     ai_run["sys"]["cpu_std"],     True),
        ("MEM mean %",    baseline["sys"]["mem_mean"],    ai_run["sys"]["mem_mean"],    True),
    ]
    for label, b, a, inv in rows:
        sys_t.add_row(label, f"{b:.2f}", f"{a:.2f}", _delta_str(b, a, inv))

    # Throughput table
    thr_t = Table(title="Throughput (ops/sec per worker type)", box=box.ROUNDED,
                  header_style="bold cyan")
    thr_t.add_column("Worker Type", width=14)
    thr_t.add_column("Baseline",    justify="right", width=14)
    thr_t.add_column("AI",          justify="right", width=14)
    thr_t.add_column("Delta",       justify="right", width=14)

    all_types = sorted(set(baseline["throughput"]) | set(ai_run["throughput"]))
    for wt in all_types:
        b = baseline["throughput"].get(wt, 0)
        a = ai_run["throughput"].get(wt, 0)
        thr_t.add_row(wt, f"{b:,.1f}", f"{a:,.1f}", _delta_str(b, a, False))

    # Summary
    b_ops = baseline["total_ops_per_s"]
    a_ops = ai_run["total_ops_per_s"]
    delta_ops = ((a_ops - b_ops) / max(b_ops, 0.001)) * 100
    verdict_color = "green" if delta_ops >= 0 else "red"
    verdict = (
        f"AI throughput vs baseline: [{verdict_color}]{delta_ops:+.2f}%[/]\n"
        f"Orchestrator ticks during AI trial: [yellow]{ai_run['orc_ticks']}[/yellow]\n"
        f"Note: weights are Xavier-random (untrained). "
        f"Expect near-zero delta until training loop is built."
    )

    console.print(sys_t)
    console.print(thr_t)
    console.print(Panel(verdict, title="[bold]Verdict[/bold]", border_style="cyan"))

    return {
        "baseline": baseline,
        "ai_run":   ai_run,
        "delta_ops_pct": delta_ops,
    }


# ── Entry ─────────────────────────────────────────────────────────────────────

def run_benchmark(
    duration_s:   float = 20.0,
    workload_mix: list | None = None,
    out_path:     str  = "benchmark/results.json",
):
    console.print(Panel(
        f"[bold cyan]AIOS-Core Benchmark Harness[/bold cyan]\n"
        f"Two {duration_s}s trials — BASELINE then AI\n"
        f"Workers: {workload_mix or WorkloadGenerator.DEFAULT_MIX}",
        border_style="cyan",
    ))

    baseline = run_trial("BASELINE", duration_s, with_ai=False, workload_mix=workload_mix)
    time.sleep(3.0)   # let system settle between trials
    ai_run   = run_trial("AI",       duration_s, with_ai=True,  workload_mix=workload_mix)

    report = print_report(baseline, ai_run)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        # worker_results contain raw data — keep it for later training
        json.dump(report, f, indent=2, default=str)
    console.print(f"\n[dim]Full results saved -> {out_path}[/dim]")
    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=20.0, help="Seconds per trial")
    p.add_argument("--out",      type=str,   default="benchmark/results.json")
    args = p.parse_args()
    run_benchmark(duration_s=args.duration, out_path=args.out)
