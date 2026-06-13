"""
AIOS-Core :: Synthetic Workload Spawner
-----------------------------------------
Spawns controlled worker processes for benchmarking:

  CPU_BURN   — pure computation (prime sieve), saturates one core
  IO_BOUND   — repeated file read/write cycles, high IO wait
  MIXED      — alternates CPU and IO in 200ms bursts
  IDLE       — mostly sleeping, occasional wakeup (background daemon sim)

Each worker reports its own throughput metric to stdout so the
benchmark engine can measure it externally.
"""

import sys
import os
import time
import math
import random
import tempfile
import argparse


# ─────────────────────────────────────────────────────────────────────────────
#  Worker implementations  (run as subprocesses)
# ─────────────────────────────────────────────────────────────────────────────

def worker_cpu_burn(duration_s: float):
    """Sieve of Eratosthenes in a tight loop. Pure CPU."""
    end   = time.perf_counter() + duration_s
    iters = 0
    while time.perf_counter() < end:
        limit = 50_000
        sieve = bytearray([1]) * limit
        for i in range(2, int(math.isqrt(limit)) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        iters += 1
    print(f"CPU_BURN  iters={iters}  ops/s={iters/duration_s:.2f}", flush=True)


def worker_io_bound(duration_s: float):
    """Repeated write + read + delete of a temp file. High IO."""
    end   = time.perf_counter() + duration_s
    iters = 0
    data  = os.urandom(256 * 1024)   # 256 KB payload
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "payload.bin")
        while time.perf_counter() < end:
            with open(path, "wb") as f:
                f.write(data)
            with open(path, "rb") as f:
                _ = f.read()
            iters += 1
    print(f"IO_BOUND  iters={iters}  ops/s={iters/duration_s:.2f}", flush=True)


def worker_mixed(duration_s: float):
    """Alternates 200ms CPU bursts with 200ms IO bursts."""
    end      = time.perf_counter() + duration_s
    cpu_ops  = 0
    io_ops   = 0
    data     = os.urandom(128 * 1024)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "payload.bin")
        while time.perf_counter() < end:
            # CPU burst
            burst_end = time.perf_counter() + 0.2
            while time.perf_counter() < burst_end and time.perf_counter() < end:
                _ = sum(i*i for i in range(5000))
                cpu_ops += 1
            # IO burst
            burst_end = time.perf_counter() + 0.2
            while time.perf_counter() < burst_end and time.perf_counter() < end:
                with open(path, "wb") as f: f.write(data)
                with open(path, "rb") as f: _ = f.read()
                io_ops += 1
    total = cpu_ops + io_ops
    print(f"MIXED  cpu_ops={cpu_ops}  io_ops={io_ops}  total_ops/s={total/duration_s:.2f}", flush=True)


def worker_idle(duration_s: float):
    """Sleeps most of the time, wakes briefly every ~500ms."""
    end    = time.perf_counter() + duration_s
    wakes  = 0
    while time.perf_counter() < end:
        time.sleep(random.uniform(0.3, 0.7))
        _ = sum(range(1000))   # tiny wakeup work
        wakes += 1
    print(f"IDLE  wakes={wakes}  wakes/s={wakes/duration_s:.2f}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point (called as subprocess)
# ─────────────────────────────────────────────────────────────────────────────

WORKERS = {
    "cpu":   worker_cpu_burn,
    "io":    worker_io_bound,
    "mixed": worker_mixed,
    "idle":  worker_idle,
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("type",     choices=list(WORKERS.keys()))
    p.add_argument("duration", type=float, help="Run time in seconds")
    args = p.parse_args()
    WORKERS[args.type](args.duration)
