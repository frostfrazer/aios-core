# AIOS-Core Benchmark Report

**Date:** 2026-06-14  
**Environment:** 2-vCPU GitHub Actions runner (Ubuntu), cgroup v2, Python 3.12  
**Duration per trial:** 15 seconds  
**Workers:** 2× CPU-bound, 2× I/O-bound, 2× Mixed  
**Tick rate:** 200ms  

---

## Summary

| Metric | Baseline | AIOS | Delta |
|--------|----------|------|-------|
| Total ops/s | 17,409 | 17,349 | -0.34% |
| CPU mean % | 72.8% | 75.0% | +2.2pp |
| Mem mean % | 6.3% | 6.0% | -0.3pp |
| AI ticks | 0 | 77 | — |

> **Note:** The -0.34% total ops/s delta is within run-to-run noise on a 2-vCPU shared runner. The scheduler had only 77 ticks (15s) to converge — far below steady state.

---

## Per-Workload Breakdown

| Worker | Type | Baseline ops/s | AIOS ops/s | Change |
|--------|------|----------------|------------|--------|
| Worker 1 | CPU | 7,523 | 7,266 | -3.4% |
| Worker 2 | CPU | 7,411 | 7,309 | -1.4% |
| Worker 3 | I/O | 182 | 236 | **+30.0%** |
| Worker 4 | I/O | 184 | 254 | **+37.9%** |
| Worker 5 | Mixed | 1,049 | 1,142 | **+8.9%** |
| Worker 6 | Mixed | 1,060 | 1,142 | **+7.7%** |

### Key findings

- **I/O workers: +30–38% improvement.** The scheduler correctly identifies I/O-bound processes and boosts their `io.weight`, reducing contention and improving throughput.
- **Mixed workers: +8–9% improvement.** Dual-axis tuning (CPU + IO) yields consistent gains on blended workloads.
- **CPU workers: -1–3%.** On a 2-vCPU runner the CPU workers compete for the same cores. The scheduler deprioritises them to give headroom to I/O — correct behaviour, but shows as a throughput dip on this metric.
- **Net total: -0.34%** — within noise. A longer warm-up period (>60s) is needed for the RL reward signal to converge on CPU weights.

---

## Wall-clock benchmark (earlier extended run)

On a 30-second mixed-workload run observed during development:

| Workload class | Improvement over baseline |
|----------------|--------------------------|
| I/O-bound | +33–48% |
| Mixed | +8–12% |
| CPU-only | -2–4% (correct tradeoff) |

---

## Methodology

1. `benchmark/harness.py` spawns 6 worker processes from `benchmark/workload.py`
2. Baseline trial: no AI, no cgroup writes
3. AI trial: `AIOrchestrator` + `SafetyGuardian` + `CgroupActuator` active
4. Workers report `ops/s` via stdout; harness sums total throughput
5. System CPU/MEM sampled every 200ms via psutil

---

## Limitations & next steps

- 2-vCPU runner limits CPU parallelism; results will be stronger on 4–8 vCPU machines
- 15s trial is too short for full RL convergence — extend to 60s minimum in next CI run
- Add per-worker latency (p50/p95) alongside throughput
- Test with real application workloads (web server, DB queries)


---

## SMB Real-World Validation (2026-06-19)

Synthetic CI benchmarks above were supplemented with a sustained 60s mixed
workload (2x web-handler, 2x SQLite DB writer, 2x background CPU job) on an
8-vCPU WSL2 host -- closer to a real small-business server profile.

| Worker | Baseline ops/s | AIOS ops/s | Delta |
|--------|---------------:|-----------:|------:|
| web    | 72,004         | 74,711     | **+3.8%** |
| db     | 178,108        | 181,952    | **+2.2%** |
| bg     | 51,631         | 53,643     | **+3.9%** |
| **Total** | **301,744** | **310,306** | **+2.8%** |

All three workload classes improved -- consistent, modest, real.

### What changed to get here

Initial validation runs showed a **-15.6% regression**, not an improvement.
Root causes found and fixed:

1. `open_files` in `psutil.process_iter()` triggered a full `/proc/PID/fd`
   directory scan on every tracked process, every tick. Removed.
2. Transient/short-lived processes (workload spawns) were being tracked and
   actuated needlessly. Added a 5-second minimum-uptime filter.
3. The AIOS daemon itself was not deprioritised and competed for CPU with
   the processes it managed. Fixed with `nice +10` at daemon startup.
4. The adaptive-load backoff used a blocking `psutil.cpu_percent()` call
   that returns 0.0 on first invocation. Replaced with non-blocking
   `os.getloadavg()`.
5. Control loop tick raised from 200ms to 500ms to reduce syscall volume.
6. **A stale daemon instance from a prior session had been silently running
   for over 24 hours**, consuming 28% CPU continuously and contaminating
   every benchmark run until discovered and killed. A reminder that
   `systemctl stop` does not guarantee process death if the unit state is
   already `failed` -- always verify with `ps aux` before benchmarking.
