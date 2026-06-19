"""
AIOS-Core :: Safety Guardian
-------------------------------
The component that makes AIOS-Core safe to run unattended on a
production server. Three independent protections:

1. RATE LIMITING — a process's weight can change by at most
   MAX_DELTA_PER_TICK fraction per tick, preventing oscillation
   and sudden starvation/boost swings.

2. ROLLBACK — tracks a rolling reward history per PID. If the
   reward trend turns sharply negative after our changes (i.e. the
   process got WORSE under AI control), the process is reset to
   default weight and QUARANTINED — the AI stops touching it for
   a cooldown period.

3. AUDIT LOG — every weight change, rollback, and quarantine event
   is appended to a JSONL file for after-the-fact review. This is
   non-negotiable for anything running on someone else's hardware.

Nothing here raises on the hot path — a guardian that crashes is a
guardian that protects nothing.
"""

import json
import time
import logging
from pathlib import Path
from collections import deque, defaultdict
from dataclasses import dataclass, field


log = logging.getLogger("safety")

DEFAULT_WEIGHT          = 0.5    # corresponds to cgroups cpu.weight=100 (neutral)
MAX_DELTA_PER_TICK      = 0.15   # max change in [0,1]-space per tick (~30 cgroup units)
REWARD_WINDOW           = 8      # ticks of reward history to consider
ROLLBACK_THRESHOLD      = -0.20  # mean reward below this -> rollback
QUARANTINE_TICKS        = 25     # ticks before a quarantined PID is re-eligible
AUDIT_LOG_PATH          = Path("logs/safety_audit.jsonl")


@dataclass
class PidState:
    current_cpu_weight: float = DEFAULT_WEIGHT
    current_io_weight:  float = DEFAULT_WEIGHT
    reward_history:     deque = field(default_factory=lambda: deque(maxlen=REWARD_WINDOW))
    quarantined_until:  int   = 0    # tick number; 0 = not quarantined
    total_rollbacks:    int   = 0


class SafetyGuardian:
    """
    Sits between the model's raw output and the actuator.

    guardian.evaluate(pid, tick, raw_cpu_w, raw_io_w, reward) -> (cpu_w, io_w, action)
      action is one of: "apply", "quarantined", "rollback"

    When action == "rollback" or "quarantined", the caller should reset
    the PID's cgroup to default weight instead of applying raw_*.
    """

    def __init__(
        self,
        max_delta:        float = MAX_DELTA_PER_TICK,
        rollback_thresh:  float = ROLLBACK_THRESHOLD,
        quarantine_ticks: int   = QUARANTINE_TICKS,
        audit_path:       Path  = None,
    ):
        self.max_delta        = max_delta
        self.rollback_thresh  = rollback_thresh
        self.quarantine_ticks = quarantine_ticks
        self.audit_path       = Path(audit_path) if audit_path else AUDIT_LOG_PATH
        self._state: dict[int, PidState] = defaultdict(PidState)

        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    # ── Core evaluation ──────────────────────────────────────────────────────

    def evaluate(
        self,
        pid:        int,
        tick:       int,
        raw_cpu_w:  float,
        raw_io_w:   float,
        reward:     float | None,
    ) -> tuple[float, float, str]:

        st = self._state[pid]

        # Record reward for trend analysis
        if reward is not None:
            st.reward_history.append(reward)

        # ── Quarantine check ──────────────────────────────────────────────────
        if tick < st.quarantined_until:
            return DEFAULT_WEIGHT, DEFAULT_WEIGHT, "quarantined"

        if tick == st.quarantined_until and st.quarantined_until > 0:
            # Quarantine just expired — clear history, give it a fresh start
            st.reward_history.clear()
            st.quarantined_until = 0
            self._audit(pid, tick, "quarantine_lifted", {})

        # ── Rollback check ──────────────────────────────────────────────────────
        if len(st.reward_history) >= REWARD_WINDOW:
            mean_reward = sum(st.reward_history) / len(st.reward_history)
            if mean_reward < self.rollback_thresh:
                st.quarantined_until = tick + self.quarantine_ticks
                st.total_rollbacks  += 1
                st.current_cpu_weight = DEFAULT_WEIGHT
                st.current_io_weight  = DEFAULT_WEIGHT
                st.reward_history.clear()
                self._audit(pid, tick, "rollback", {
                    "mean_reward": round(mean_reward, 4),
                    "threshold":   self.rollback_thresh,
                    "quarantine_until": st.quarantined_until,
                    "total_rollbacks": st.total_rollbacks,
                })
                import sys as _sys
                print(
                    f"[AIOS ALERT] PID {pid} quarantined at tick {tick} "
                    f"(mean_reward={mean_reward:.4f} < {self.rollback_thresh}, "
                    f"rollbacks={st.total_rollbacks}, resumes_tick={st.quarantined_until})",
                    file=_sys.stderr, flush=True,
                )
                return DEFAULT_WEIGHT, DEFAULT_WEIGHT, "rollback"

        # ── Rate-limited apply ──────────────────────────────────────────────────
        new_cpu = self._step_towards(st.current_cpu_weight, raw_cpu_w)
        new_io  = self._step_towards(st.current_io_weight,  raw_io_w)

        if abs(new_cpu - st.current_cpu_weight) > 0.005 or abs(new_io - st.current_io_weight) > 0.005:
            self._audit(pid, tick, "apply", {
                "cpu_weight": round(new_cpu, 4),
                "io_weight":  round(new_io, 4),
                "raw_cpu":    round(raw_cpu_w, 4),
                "raw_io":     round(raw_io_w, 4),
            })

        st.current_cpu_weight = new_cpu
        st.current_io_weight  = new_io
        return new_cpu, new_io, "apply"

    def _step_towards(self, current: float, target: float) -> float:
        delta = target - current
        if abs(delta) > self.max_delta:
            delta = self.max_delta if delta > 0 else -self.max_delta
        return max(0.0, min(1.0, current + delta))

    # ── Introspection ─────────────────────────────────────────────────────────

    def is_quarantined(self, pid: int, tick: int) -> bool:
        return tick < self._state[pid].quarantined_until

    def stats(self) -> dict:
        total_rb = sum(s.total_rollbacks for s in self._state.values())
        quarantined = sum(1 for s in self._state.values() if s.total_rollbacks > 0)
        return {
            "tracked_pids":    len(self._state),
            "total_rollbacks": total_rb,
            "ever_quarantined": quarantined,
        }

    def forget(self, pid: int):
        """Drop state for a PID that has exited."""
        self._state.pop(pid, None)

    # ── Audit log ─────────────────────────────────────────────────────────────

    def _audit(self, pid: int, tick: int, event: str, extra: dict):
        # In-process size guard -- don't rely solely on cron-driven logrotate
        try:
            import os as _os
            if _os.path.exists(self.audit_path) and _os.path.getsize(self.audit_path) > 50 * 1024 * 1024:
                _os.rename(self.audit_path, self.audit_path + ".1")
        except OSError:
            pass

        record = {
            "ts":    time.time(),
            "tick":  tick,
            "pid":   pid,
            "event": event,
            **extra,
        }
        try:
            with open(self.audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            log.debug(f"audit write failed: {e}")
