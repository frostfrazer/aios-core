"""
AIOS-Core :: Telemetry Collector
----------------------------------
Reads real process telemetry from the OS via psutil, normalises it
into a 12-dim feature vector ready for the NeuralScheduler.

Feature vector layout (all values normalised to [0,1] or [-1,1]):
  [0]  cpu_pct          — process CPU %  / 100
  [1]  mem_rss_norm     — RSS bytes / 4GB
  [2]  io_read_rate     — bytes/s read  (log-scaled)
  [3]  io_write_rate    — bytes/s write (log-scaled)
  [4]  ctx_vol_rate     — voluntary   context switches / s
  [5]  ctx_nvol_rate    — involuntary context switches / s
  [6]  thread_count     — num threads  / 64
  [7]  open_files_norm  — open fd count / 256
  [8]  status_encoded   — {running:1, sleeping:0.5, disk-wait:-0.5, zombie:-1}
  [9]  cpu_affinity_frac— pinned cores / total cores
  [10] nice_norm        — nice value mapped to [-1, 1]  (linux: -20..19)
  [11] age_norm         — process age seconds (log-scaled, cap 1hr)
"""

import time
import math
import psutil
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


_STATUS_MAP = {
    "running":  1.0,
    "sleeping": 0.5,
    "disk-sleep": -0.5,
    "stopped":  -0.8,
    "zombie":   -1.0,
}

_LOG_SCALE_CAP = 1e9   # 1 GB/s ceiling for IO rates
_TOTAL_CORES   = psutil.cpu_count(logical=True) or 1


@dataclass
class ProcessSnapshot:
    pid:            int
    cpu_pct:        float = 0.0
    mem_rss:        int   = 0
    io_read_bps:    float = 0.0
    io_write_bps:   float = 0.0
    ctx_vol:        int   = 0
    ctx_nvol:       int   = 0
    threads:        int   = 1
    open_files:     int   = 0
    status:         str   = "sleeping"
    affinity_cores: int   = _TOTAL_CORES
    nice:           int   = 0
    create_time:    float = field(default_factory=time.time)

    _prev_io_read:  int   = 0
    _prev_io_write: int   = 0
    _prev_ctx_vol:  int   = 0
    _prev_ctx_nvol: int   = 0
    _prev_ts:       float = field(default_factory=time.time)


def _log_norm(value: float, cap: float = _LOG_SCALE_CAP) -> float:
    """Log-scale a bytes/s value into [0, 1]."""
    if value <= 0:
        return 0.0
    return min(math.log1p(value) / math.log1p(cap), 1.0)


class TelemetryCollector:
    """
    Collects and normalises per-process telemetry into feature vectors.
    Maintains per-process state to compute delta rates between ticks.
    """

    FEATURE_DIM = 12

    def __init__(self, window_ms: int = 100):
        self.window_ms  = window_ms
        self._snapshots: dict[int, ProcessSnapshot] = {}
        self._proc_cache: dict[int, psutil.Process] = {}
        self._last_tick = time.time()

    # ── Public API ────────────────────────────────────────────────────────────

    def collect_all(self) -> dict[int, np.ndarray]:
        """
        Snapshot all accessible processes and return {pid: feature_vector}.
        Inaccessible processes (permissions, zombies) are silently skipped.
        """
        now     = time.time()
        dt      = max(now - self._last_tick, 0.001)
        results = {}

        for proc in psutil.process_iter(
            ["pid", "cpu_percent", "memory_info", "io_counters",
             "num_ctx_switches", "num_threads",
             "status", "cpu_affinity", "nice", "create_time"]
        ):
            try:
                pid  = proc.info["pid"]
                # Skip processes younger than 3s — transient, not worth tuning
                create_time = proc.info.get("create_time") or 0
                if (now - create_time) < 5.0:
                    continue
                feat = self._extract(proc.info, pid, dt)
                if feat is not None:
                    results[pid] = feat
            except (psutil.NoSuchProcess, psutil.AccessDenied,
                    psutil.ZombieProcess):
                continue

        self._last_tick = now
        return results

    def collect_pid(self, pid: int) -> Optional[np.ndarray]:
        """Collect features for a single PID. Caches psutil.Process objects
        across calls so cpu_percent() deltas and lookups stay cheap."""
        try:
            proc = self._proc_cache.get(pid)
            if proc is None:
                proc = psutil.Process(pid)
                proc.cpu_percent(interval=None)   # prime the internal timer
                self._proc_cache[pid] = proc

            info = proc.as_dict(
                attrs=["pid", "cpu_percent", "memory_info", "io_counters",
                       "num_ctx_switches", "num_threads",
                       "status", "cpu_affinity", "nice", "create_time"]
            )
            info["open_files"] = []   # skip expensive /proc/pid/fd walk
            dt = max(time.time() - self._last_tick, 0.001)
            return self._extract(info, pid, dt)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self._proc_cache.pop(pid, None)
            return None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _extract(self, info: dict, pid: int, dt: float) -> Optional[np.ndarray]:
        snap = self._snapshots.get(pid, ProcessSnapshot(pid=pid))

        # IO deltas
        io = info.get("io_counters")
        io_read_bps  = 0.0
        io_write_bps = 0.0
        if io:
            io_read_bps  = max(io.read_bytes  - snap._prev_io_read,  0) / dt
            io_write_bps = max(io.write_bytes - snap._prev_io_write, 0) / dt
            snap._prev_io_read  = io.read_bytes
            snap._prev_io_write = io.write_bytes

        # Context switch deltas
        ctx = info.get("num_ctx_switches")
        ctx_vol_rate  = 0.0
        ctx_nvol_rate = 0.0
        if ctx:
            ctx_vol_rate  = max(ctx.voluntary   - snap._prev_ctx_vol,  0) / dt
            ctx_nvol_rate = max(ctx.involuntary - snap._prev_ctx_nvol, 0) / dt
            snap._prev_ctx_vol  = ctx.voluntary
            snap._prev_ctx_nvol = ctx.involuntary

        mem_rss  = info["memory_info"].rss if info.get("memory_info") else 0
        threads  = info.get("num_threads") or 1
        status   = info.get("status", "sleeping")
        nice_val = info.get("nice") or 0
        aff      = len(info["cpu_affinity"]) if info.get("cpu_affinity") else _TOTAL_CORES
        ctime    = info.get("create_time") or time.time()
        age_s    = max(time.time() - ctime, 0)

        # Save snapshot
        snap.cpu_pct        = info.get("cpu_percent") or 0.0
        snap.mem_rss        = mem_rss
        snap.io_read_bps    = io_read_bps
        snap.io_write_bps   = io_write_bps
        snap.threads        = threads
        snap.status         = status
        snap.affinity_cores = aff
        snap.nice           = nice_val
        snap.create_time    = ctime
        self._snapshots[pid] = snap

        # Build normalised feature vector
        vec = np.array([
            min(snap.cpu_pct / 100.0, 1.0),                         # [0]
            min(mem_rss / (4 * 1024**3), 1.0),                      # [1]
            _log_norm(io_read_bps),                                  # [2]
            _log_norm(io_write_bps),                                 # [3]
            _log_norm(ctx_vol_rate,  cap=1000),                      # [4]
            _log_norm(ctx_nvol_rate, cap=1000),                      # [5]
            min(threads / 64.0, 1.0),                                # [6]
            min(len(info.get("open_files") or []) / 256.0, 1.0),    # [7]
            _STATUS_MAP.get(status, 0.0),                            # [8]
            aff / _TOTAL_CORES,                                      # [9]
            (nice_val + 20) / 39.0 * 2 - 1,                         # [10] → [-1,1]
            _log_norm(age_s, cap=3600),                              # [11]
        ], dtype=np.float32)

        return vec
