"""
AIOS-Core :: Workload Generator
---------------------------------
Spawns controlled worker processes that simulate real OS workload patterns:

  CPU_BURN    — tight compute loop (prime sieve), 100% CPU bound
  IO_THRASH   — rapid small file read/write, IO bound
  MIXED       — alternates CPU and IO phases every 200ms
  IDLE        — sleeps with occasional wakeups (background daemon sim)

Each worker reports its own throughput (ops/sec) to a shared result queue,
so the benchmark harness can measure actual useful work done — not just
wall-clock time.
"""

import multiprocessing as mp
import time
import os
import math
import tempfile
import random
import sys


# ── Worker functions (must be top-level for mp.Process pickling) ──────────────

def _cpu_burn_worker(result_q: mp.Queue, stop_evt: mp.Event, worker_id: int):
    """Prime sieve in a tight loop. Reports primes found per second."""
    ops = 0
    t0  = time.perf_counter()

    def sieve(n):
        # Sieve of Eratosthenes up to n
        bits = bytearray([1]) * (n + 1)
        bits[0] = bits[1] = 0
        for i in range(2, int(n**0.5) + 1):
            if bits[i]:
                bits[i*i::i] = bytearray(len(bits[i*i::i]))
        return sum(bits)

    while not stop_evt.is_set():
        count = sieve(10_000)
        ops  += count

    elapsed = time.perf_counter() - t0
    result_q.put({
        "worker_id": worker_id,
        "type":      "CPU_BURN",
        "ops":       ops,
        "elapsed":   elapsed,
        "ops_per_s": ops / max(elapsed, 0.001),
    })


def _io_thrash_worker(result_q: mp.Queue, stop_evt: mp.Event, worker_id: int):
    """Write and read back 4KB blocks in a temp file. Reports MB/s."""
    ops       = 0
    bytes_rw  = 0
    t0        = time.perf_counter()
    block     = os.urandom(4096)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".aios_bench") as f:
        fpath = f.name

    try:
        while not stop_evt.is_set():
            with open(fpath, "wb") as f:
                f.write(block)
            with open(fpath, "rb") as f:
                _ = f.read()
            ops      += 1
            bytes_rw += len(block) * 2
    finally:
        try:
            os.unlink(fpath)
        except OSError:
            pass

    elapsed = time.perf_counter() - t0
    result_q.put({
        "worker_id": worker_id,
        "type":      "IO_THRASH",
        "ops":       ops,
        "elapsed":   elapsed,
        "ops_per_s": ops / max(elapsed, 0.001),
        "mb_per_s":  bytes_rw / max(elapsed, 0.001) / 1024**2,
    })


def _mixed_worker(result_q: mp.Queue, stop_evt: mp.Event, worker_id: int):
    """Alternates 200ms CPU bursts and 200ms IO bursts."""
    ops      = 0
    t0       = time.perf_counter()
    phase_t  = time.perf_counter()
    phase    = "cpu"
    block    = os.urandom(4096)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".aios_mix") as f:
        fpath = f.name

    try:
        while not stop_evt.is_set():
            now = time.perf_counter()
            if now - phase_t > 0.2:
                phase   = "io" if phase == "cpu" else "cpu"
                phase_t = now

            if phase == "cpu":
                # mini compute burst
                _ = sum(i * i for i in range(500))
                ops += 1
            else:
                # mini IO burst
                with open(fpath, "wb") as f:
                    f.write(block)
                ops += 1
    finally:
        try:
            os.unlink(fpath)
        except OSError:
            pass

    elapsed = time.perf_counter() - t0
    result_q.put({
        "worker_id": worker_id,
        "type":      "MIXED",
        "ops":       ops,
        "elapsed":   elapsed,
        "ops_per_s": ops / max(elapsed, 0.001),
    })


def _idle_worker(result_q: mp.Queue, stop_evt: mp.Event, worker_id: int):
    """Sleeps most of the time, wakes occasionally — background daemon sim."""
    wakeups = 0
    t0      = time.perf_counter()

    while not stop_evt.is_set():
        time.sleep(random.uniform(0.05, 0.15))
        _ = sum(range(100))   # tiny wakeup computation
        wakeups += 1

    elapsed = time.perf_counter() - t0
    result_q.put({
        "worker_id": worker_id,
        "type":      "IDLE",
        "ops":       wakeups,
        "elapsed":   elapsed,
        "ops_per_s": wakeups / max(elapsed, 0.001),
    })


# ── Workload manager ──────────────────────────────────────────────────────────

_WORKER_FNS = {
    "CPU_BURN": _cpu_burn_worker,
    "IO_THRASH": _io_thrash_worker,
    "MIXED":    _mixed_worker,
    "IDLE":     _idle_worker,
}

class WorkloadGenerator:
    """
    Spawns a configurable mix of worker processes and collects their results.

    Default mix (mirrors a "developer workstation" profile):
      2x CPU_BURN  (compilation-like)
      2x IO_THRASH (file-system-heavy)
      2x MIXED     (web server-like)
      2x IDLE      (background services)
    """

    DEFAULT_MIX = [
        ("CPU_BURN",  2),
        ("IO_THRASH", 2),
        ("MIXED",     2),
        ("IDLE",      2),
    ]

    def __init__(self, mix=None):
        self.mix      = mix or self.DEFAULT_MIX
        self._procs:  list[mp.Process]  = []
        self._result_q = mp.Queue()
        self._stop_evt = mp.Event()

    def start(self):
        """Spawn all workers."""
        wid = 0
        for wtype, count in self.mix:
            fn = _WORKER_FNS[wtype]
            for _ in range(count):
                p = mp.Process(
                    target=fn,
                    args=(self._result_q, self._stop_evt, wid),
                    daemon=True,
                )
                p.start()
                self._procs.append(p)
                wid += 1
        return [p.pid for p in self._procs]

    def stop(self) -> list[dict]:
        """Signal workers to stop and collect results."""
        self._stop_evt.set()
        results = []
        deadline = time.time() + 5.0
        remaining = len(self._procs)
        while remaining > 0 and time.time() < deadline:
            try:
                r = self._result_q.get(timeout=0.5)
                results.append(r)
                remaining -= 1
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
        self._procs.clear()
        return results

    @property
    def worker_count(self) -> int:
        return len(self._procs)
