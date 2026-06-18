#!/usr/bin/env python3
"""
AIOS-Core :: Daemon Entry Point
---------------------------------
Runs as a systemd service. Reads config from /etc/aios/config.json,
writes PID file, logs to journald via stderr (systemd captures it),
and handles SIGTERM/SIGHUP cleanly.

Install via: sudo aios install
"""

import sys
import os
import signal
import logging
import json
import time
from pathlib import Path

# Allow running from repo root or installed path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from orchestrator.engine import AIOrchestrator

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "tick_ms":       200,
    "apply_os":      True,
    "model_path":    "/etc/aios/weights.json",
    "log_level":     "INFO",
    "pid_file":      "/run/aios/aios.pid",
    "audit_log":     "/var/log/aios/audit.jsonl",
}

CONFIG_PATH = Path(os.environ.get("AIOS_CONFIG", "/etc/aios/config.json"))


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text()))
        except Exception as e:
            print(f"[aios] Warning: could not parse config {CONFIG_PATH}: {e}", file=sys.stderr)
    return cfg


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(level: str):
    # Log to stderr -- systemd/journald captures it automatically.
    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )


# ── PID file ──────────────────────────────────────────────────────────────────

def write_pid(path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(os.getpid()))


def remove_pid(path: str):
    try:
        Path(path).unlink()
    except OSError:
        pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Renice AIOS to low priority — it must never compete with managed processes
    try:
        os.nice(10)
    except OSError:
        pass
    cfg = load_config()
    setup_logging(cfg["log_level"])
    log = logging.getLogger("daemon")

    log.info(f"AIOS-Core daemon starting (pid={os.getpid()})")
    log.info(f"Config: tick={cfg['tick_ms']}ms  apply_os={cfg['apply_os']}  "
             f"model={cfg['model_path']}")

    write_pid(cfg["pid_file"])

    orc = AIOrchestrator(
        tick_ms    = cfg["tick_ms"],
        apply_os   = cfg["apply_os"],
        model_path = cfg["model_path"] if Path(cfg["model_path"]).exists() else None,
        audit_log  = cfg["audit_log"],
    )

    # ── Signal handlers ───────────────────────────────────────────────────────
    def _shutdown(sig, frame):
        log.info(f"Received signal {sig}, shutting down...")
        orc.stop()
        remove_pid(cfg["pid_file"])
        log.info(f"AIOS-Core stopped. Ticks: {orc.tick_count}  "
                 f"Guardian: {orc.guardian.stats()}")
        sys.exit(0)

    def _reload(sig, frame):
        """SIGHUP: reload model weights without restarting."""
        model_path = cfg.get("model_path")
        if model_path and Path(model_path).exists():
            log.info(f"SIGHUP: reloading model weights from {model_path}")
            orc.model.load(model_path)
        else:
            log.warning("SIGHUP: no model path configured, ignoring.")

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGHUP,  _reload)

    orc.start()
    log.info("AIOS-Core daemon running.")

    # Keep main thread alive -- orchestrator runs in background thread.
    while True:
        time.sleep(60)
        log.info(f"Heartbeat: ticks={orc.tick_count}  "
                 f"guardian={orc.guardian.stats()}")


if __name__ == "__main__":
    main()
