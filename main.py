"""
AIOS-Core :: Entry Point
--------------------------
Usage:
    python main.py                   # observe mode, no OS changes
    python main.py --apply           # actually adjust process priorities
    python main.py --tick 100        # 100ms control loop
    python main.py --save weights/   # auto-save model every 60s
    python main.py --headless        # no dashboard, just log to stdout

Ctrl+C to stop — model weights saved automatically on exit.
"""

import sys
import os
import time
import signal
import logging
import argparse
import threading
from pathlib import Path

# ── Make sure project root is on sys.path ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from orchestrator.engine import AIOrchestrator
from simulator.dashboard import Dashboard


# ── Logging (file only — stdout reserved for Rich) ────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    filename="logs/aios.log",
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("main")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="AIOS-Core Orchestrator")
    p.add_argument("--tick",     type=int,  default=200,   help="Control loop period (ms)")
    p.add_argument("--apply",    action="store_true",      help="Apply OS nice() adjustments")
    p.add_argument("--save",     type=str,  default="weights", help="Directory for model checkpoints")
    p.add_argument("--load",     type=str,  default=None,  help="Load model weights from file")
    p.add_argument("--headless", action="store_true",      help="Disable dashboard")
    p.add_argument("--autosave", type=int,  default=60,    help="Auto-save interval (seconds, 0=off)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args      = parse_args()
    stop_event = threading.Event()

    # Graceful shutdown on Ctrl+C / SIGTERM
    def _shutdown(sig=None, frame=None):
        print("\n[AIOS] Shutting down…")
        stop_event.set()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Orchestrator ──────────────────────────────────────────────────────────
    orc = AIOrchestrator(
        tick_ms   = args.tick,
        apply_os  = args.apply,
        model_path= args.load,
    )

    # ── Dashboard ─────────────────────────────────────────────────────────────
    dash = None
    if not args.headless:
        dash = Dashboard(refresh_ms=max(args.tick, 200))
        orc.register_callback(dash.on_tick)
    else:
        # Simple stdout logger callback
        def _log_cb(tick, decisions):
            if tick % 10 == 0:
                print(f"[tick {tick:06d}]  procs={len(decisions)}")
        orc.register_callback(_log_cb)

    # ── Auto-save thread ──────────────────────────────────────────────────────
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    def _autosave():
        while not stop_event.is_set():
            time.sleep(args.autosave)
            if stop_event.is_set():
                break
            path = save_dir / f"weights_tick{orc.tick_count:06d}.json"
            orc.save_model(str(path))

    if args.autosave > 0:
        threading.Thread(target=_autosave, daemon=True).start()

    # ── Boot ──────────────────────────────────────────────────────────────────
    print(f"[AIOS] Starting orchestrator — tick={args.tick}ms  "
          f"apply_os={args.apply}  headless={args.headless}")
    orc.start()

    if dash:
        dash.run(stop_event)   # blocks until Ctrl+C
    else:
        stop_event.wait()

    # ── Shutdown ──────────────────────────────────────────────────────────────
    orc.stop()
    final_path = save_dir / "weights_final.json"
    orc.save_model(str(final_path))
    print(f"[AIOS] Model saved → {final_path}")
    print(f"[AIOS] Total ticks: {orc.tick_count}")


if __name__ == "__main__":
    main()
