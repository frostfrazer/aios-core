"""
AIOS-Core :: AI Orchestrator
------------------------------
The closed-loop control engine. Every tick it:
  1. Pulls telemetry from all running processes
  2. Runs each process's features through NeuralScheduler
  3. Emits tuning decisions (logged; on Linux this would write to /proc)
  4. Optionally applies OS-level nice() adjustment on Windows via psutil
  5. Computes a reward signal from subsequent CPU efficiency
  6. Applies online weight update (RL-lite)

This is the module that maps 1-to-1 with the kernel AI capsule in
the seL4 design — isolate, observe, decide, actuate.
"""

import time
import threading
import logging
import psutil
import numpy as np
from collections import deque
from typing import Optional

from models.neural_scheduler import NeuralScheduler
from telemetry.collector import TelemetryCollector


log = logging.getLogger("orchestrator")


# ── Decision record ────────────────────────────────────────────────────────────

class Decision:
    __slots__ = ("pid", "ts", "features", "params", "reward")

    def __init__(self, pid, ts, features, params):
        self.pid      = pid
        self.ts       = ts
        self.features = features
        self.params   = params
        self.reward   = None       # filled in on next tick


# ── Main orchestrator ─────────────────────────────────────────────────────────

class AIOrchestrator:
    """
    Core control loop. Designed to be a drop-in for the seL4 AI capsule.

    tick_ms  : control loop period (mirrors FEATURE_WINDOW_MS in the C module)
    apply_os : if True, call psutil.Process.nice() to actually adjust priority
               (Windows: requires admin for some processes — fails gracefully)
    """

    DECISION_HISTORY = 500   # rolling window for reward computation

    def __init__(
        self,
        tick_ms:    int  = 200,
        apply_os:   bool = False,
        model_path: Optional[str]  = None,
        watch_pids: Optional[list] = None,
    ):
        self.tick_ms     = tick_ms
        self.apply_os    = apply_os
        self.watch_pids  = watch_pids   # if set, only observe these PIDs
        self.model      = NeuralScheduler()
        self.collector  = TelemetryCollector(window_ms=tick_ms)
        self._history:  dict[int, deque] = {}   # pid → deque[Decision]
        self._prev_cpu: dict[int, float] = {}   # for reward computation
        self._running   = False
        self._thread:   Optional[threading.Thread] = None
        self._tick_count = 0
        self.online_update = True   # if False, skip apply_reward (eval mode)
        self._callbacks = []   # list of (fn) called after each tick
        self._started   = threading.Event()   # set after first tick begins

        if model_path:
            self.model.load(model_path)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=False)
        self._thread.start()
        self._started.wait(timeout=5.0)   # block until first tick confirms loop is live
        log.info(f"Orchestrator started -- tick={self.tick_ms}ms  apply_os={self.apply_os}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        log.info(f"Orchestrator stopped after {self._tick_count} ticks.")

    def register_callback(self, fn):
        """fn(tick_num, decisions: dict[pid, Decision]) called each tick."""
        self._callbacks.append(fn)

    # ── Core loop ─────────────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            t0 = time.perf_counter()
            try:
                self._tick()
            except Exception as e:
                log.warning(f"Tick error: {e}")
            elapsed = (time.perf_counter() - t0) * 1000
            sleep_ms = max(0, self.tick_ms - elapsed)
            time.sleep(sleep_ms / 1000.0)

    def _tick(self):
        self._tick_count += 1
        if self._tick_count == 1:
            self._started.set()   # unblock start()
        ts = time.time()

        # Focused collection (benchmark) vs full-system scan (dashboard)
        if self.watch_pids:
            features = {}
            for pid in self.watch_pids:
                f = self.collector.collect_pid(pid)
                if f is not None:
                    features[pid] = f
        else:
            features = self.collector.collect_all()
        decisions = {}

        for pid, feat in features.items():
            params = self.model.predict(feat)
            dec    = Decision(pid, ts, feat, params)
            decisions[pid] = dec

            # Store decision for reward attribution
            if pid not in self._history:
                self._history[pid] = deque(maxlen=self.DECISION_HISTORY)
            self._history[pid].append(dec)

            # Apply OS-level nice adjustment (optional, graceful failure)
            if self.apply_os and params["nice_delta"] != 0:
                self._apply_nice(pid, params["nice_delta"])

        # Compute rewards for previous tick's decisions
        self._compute_rewards(features)

        # Fire callbacks (UI, logger, etc.)
        for cb in self._callbacks:
            try:
                cb(self._tick_count, decisions)
            except Exception:
                pass

    # ── OS actuation ──────────────────────────────────────────────────────────

    def _apply_nice(self, pid: int, delta: int):
        """Nudge a process's OS priority. Fails silently on access denial."""
        try:
            proc     = psutil.Process(pid)
            current  = proc.nice()
            # Windows nice maps to ABOVE_NORMAL_PRIORITY_CLASS etc.
            # psutil uses -20..19 scale on Windows too.
            new_nice = max(-20, min(19, current + delta))
            if new_nice != current:
                proc.nice(new_nice)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            pass

    # ── Reward signal ─────────────────────────────────────────────────────────

    def _compute_rewards(self, current_features: dict[int, np.ndarray]):
        """
        Reward = improvement in CPU efficiency vs baseline.
        Crude heuristic: if process CPU% dropped while still running → good.
        This will be replaced with a proper performance counter in Phase 2.
        """
        for pid, feat in current_features.items():
            current_cpu = float(feat[0])   # feature[0] = normalised cpu_pct
            prev_cpu    = self._prev_cpu.get(pid, current_cpu)

            # Positive reward if CPU went down (less thrashing), negative if up
            reward = float(np.clip(prev_cpu - current_cpu, -0.5, 0.5))

            # Attribute reward to last decision for this PID
            hist = self._history.get(pid)
            if hist and hist[-1].reward is None:
                hist[-1].reward = reward
                if self.online_update:
                    self.model.apply_reward(hist[-1].features, reward)

            self._prev_cpu[pid] = current_cpu

    # ── Introspection ─────────────────────────────────────────────────────────

    def get_recent_decisions(self, pid: int, n: int = 10) -> list:
        hist = self._history.get(pid, deque())
        return list(hist)[-n:]

    def save_model(self, path: str):
        self.model.save(path)

    @property
    def tick_count(self) -> int:
        return self._tick_count
