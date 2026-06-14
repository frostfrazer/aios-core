"""
AIOS-Core :: Cgroup Actuator
-------------------------------
Linux-only. Applies AI tuning decisions via cgroups v2 (cpu.weight,
optionally io.weight if BFQ is active) instead of the coarse, often
ineffective nice() syscall.

Each watched PID gets its own leaf cgroup under /sys/fs/cgroup/aios/.
cpu.weight range is [1, 10000], default 100 — we map the model's
cpu_weight output [0,1] to roughly [1, 200] so 0.5 == default (100),
giving the AI headroom to boost (up to 2x default) or deprioritise
(down to near-zero) without ever fully starving a process.

All cgroup operations are best-effort: if the kernel/permissions don't
support a write, we log once and continue. This must NEVER raise and
crash the orchestrator — a misbehaving actuator is worse than no
actuator.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional


log = logging.getLogger("actuator")

CGROUP_ROOT = Path("/sys/fs/cgroup/aios")
SYSTEM_ROOT_PROCS = Path("/sys/fs/cgroup/cgroup.procs")   # true root -- exempt from
                                                            # the "no internal process"
                                                            # constraint, safe to move
                                                            # PIDs back to.

# cpu.weight mapping: model output 0.5 (neutral) -> cgroups default 100
CPU_WEIGHT_SCALE = 200     # weight = clamp(round(cpu_weight * SCALE), MIN, MAX)
CPU_WEIGHT_MIN   = 1
CPU_WEIGHT_MAX   = 1000    # cap well below the 10000 ceiling — see safety.py

IO_WEIGHT_SCALE  = 200
IO_WEIGHT_MIN    = 1
IO_WEIGHT_MAX    = 1000


def _write(path: Path, value: str) -> bool:
    try:
        path.write_text(str(value))
        return True
    except (OSError, PermissionError) as e:
        log.debug(f"write failed {path}: {e}")
        return False


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except (OSError, FileNotFoundError):
        return None


class CgroupActuator:
    """
    Manages per-PID cgroups for AI-driven resource weighting.

    Usage:
        act = CgroupActuator()
        act.apply(pid, cpu_weight=0.8, io_weight=0.3)
        ...
        act.reset(pid)          # return to default weight, keep cgroup
        act.cleanup_all()        # remove all AIOS cgroups (on shutdown)
    """

    def __init__(self, root: Path = CGROUP_ROOT):
        self.root = root
        self.available = False
        self.io_weight_available = False
        self._managed_pids: set[int] = set()
        self._init_root()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _init_root(self):
        if sys.platform != "linux":
            log.info("CgroupActuator: non-Linux platform, disabled.")
            return

        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            log.warning(f"CgroupActuator: cannot create {self.root}: {e}. "
                        f"Likely needs root or cgroup delegation. Disabled.")
            return

        # Enable cpu controller for our subtree (hop 1: root -> aios)
        root_subtree = Path("/sys/fs/cgroup/cgroup.subtree_control")
        current = _read(root_subtree) or ""
        if "cpu" not in current.split():
            _write(root_subtree, "+cpu")
        if "io" not in current.split():
            _write(root_subtree, "+io")

        # Hop 2: aios -> pid_N. Controllers must be enabled at EACH level
        # of the cgroup v2 hierarchy, not just the top.
        aios_controllers = _read(self.root / "cgroup.controllers") or ""
        aios_subtree     = self.root / "cgroup.subtree_control"
        aios_subtree_cur = _read(aios_subtree) or ""

        if "cpu" in aios_controllers.split() and "cpu" not in aios_subtree_cur.split():
            _write(aios_subtree, "+cpu")
        if "io" in aios_controllers.split() and "io" not in aios_subtree_cur.split():
            _write(aios_subtree, "+io")

        # Probe io.weight availability (requires BFQ on the underlying device)
        probe_dir = self.root / "_probe"
        try:
            probe_dir.mkdir(exist_ok=True)
            self.io_weight_available = (probe_dir / "io.weight").exists()
            probe_dir.rmdir()
        except (OSError, PermissionError):
            pass

        self.available = (self.root / "cgroup.procs").exists() or self.root.exists()
        log.info(f"CgroupActuator: ready. io.weight_available={self.io_weight_available}")

    # ── Per-PID cgroup management ────────────────────────────────────────────

    def _pid_dir(self, pid: int) -> Path:
        return self.root / f"pid_{pid}"

    def _ensure_pid_cgroup(self, pid: int) -> Optional[Path]:
        d = self._pid_dir(pid)
        if pid in self._managed_pids:
            return d   # already created and migrated -- skip redundant writes
        if not d.exists():
            try:
                d.mkdir(exist_ok=True)
            except (OSError, PermissionError) as e:
                log.debug(f"cannot create cgroup for pid {pid}: {e}")
                return None
        # Move the process in ONCE. Re-writing cgroup.procs every tick
        # for a PID already resident triggers cgroup_mutex contention and
        # disrupts CPU cache locality for tight loops -- measured ~19%
        # throughput loss when done at 200ms intervals across 16 PIDs.
        if not _write(d / "cgroup.procs", str(pid)):
            return None
        self._managed_pids.add(pid)
        return d

    # ── Public API ────────────────────────────────────────────────────────────

    def apply(self, pid: int, cpu_weight: float, io_weight: float) -> bool:
        """
        cpu_weight, io_weight: model outputs in [0, 1].
        Returns True if at least the CPU weight was applied.
        """
        if not self.available:
            return False

        d = self._ensure_pid_cgroup(pid)
        if d is None:
            return False

        cw = self._clamp(round(cpu_weight * CPU_WEIGHT_SCALE), CPU_WEIGHT_MIN, CPU_WEIGHT_MAX)
        ok = _write(d / "cpu.weight", cw)

        if self.io_weight_available:
            iw = self._clamp(round(io_weight * IO_WEIGHT_SCALE), IO_WEIGHT_MIN, IO_WEIGHT_MAX)
            _write(d / "io.weight", iw)

        return ok

    def reset(self, pid: int):
        """Return a PID's cgroup to default weight (100) without removing it."""
        d = self._pid_dir(pid)
        if d.exists():
            _write(d / "cpu.weight", 100)
            if self.io_weight_available:
                _write(d / "io.weight", 100)

    def remove(self, pid: int):
        """Move PID back to the system root cgroup and remove its leaf cgroup."""
        d = self._pid_dir(pid)
        if not d.exists():
            self._managed_pids.discard(pid)
            return
        # Move process back to the TRUE system root (exempt from the cgroup v2
        # "no internal process" constraint) before rmdir.
        _write(SYSTEM_ROOT_PROCS, str(pid))
        try:
            d.rmdir()
        except OSError:
            pass   # process may have exited already; cgroup auto-cleans
        self._managed_pids.discard(pid)

    def cleanup_all(self):
        """Called on orchestrator shutdown — remove every AIOS cgroup."""
        for pid in list(self._managed_pids):
            self.remove(pid)
        # Best-effort: remove any stale dirs left from a previous crashed run
        if self.root.exists():
            for child in self.root.iterdir():
                if child.is_dir() and child.name.startswith("pid_"):
                    try:
                        for line in (_read(child / "cgroup.procs") or "").splitlines():
                            if line.strip():
                                _write(SYSTEM_ROOT_PROCS, line.strip())
                        child.rmdir()
                    except OSError:
                        pass

    @staticmethod
    def _clamp(value: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, int(value)))
