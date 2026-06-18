#!/usr/bin/env python3
"""
AIOS-Core :: CLI
------------------
Commands:
  aios status       -- daemon health + guardian stats
  aios top          -- live per-process decisions (like htop, but AI-aware)
  aios log          -- tail the audit log
  aios stop         -- graceful shutdown
  aios reload       -- hot-reload model weights (SIGHUP)
  aios version      -- print version

Usage:
  python3 cli.py <command>
  # or via /usr/bin/aios symlink after install
"""

import sys
import os
import signal
import json
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

VERSION = "0.1.0"
PID_FILE = Path(os.environ.get("AIOS_PID_FILE", "/run/aios/aios.pid"))
AUDIT_LOG = Path(os.environ.get("AIOS_AUDIT_LOG", "/var/log/aios/audit.jsonl"))
CONFIG_PATH = Path(os.environ.get("AIOS_CONFIG", "/etc/aios/config.json"))

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_pid() -> int | None:
    try:
        return int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None

def _daemon_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False

def _red(s):    return f"\033[31m{s}\033[0m"
def _green(s):  return f"\033[32m{s}\033[0m"
def _yellow(s): return f"\033[33m{s}\033[0m"
def _cyan(s):   return f"\033[36m{s}\033[0m"
def _bold(s):   return f"\033[1m{s}\033[0m"
def _dim(s):    return f"\033[2m{s}\033[0m"

def _bar(v: float, w: int = 10) -> str:
    filled = int(round(max(0.0, min(1.0, v)) * w))
    return _cyan("█" * filled) + _dim("░" * (w - filled))


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_version():
    print(f"aios-core {VERSION}")


def cmd_status():
    pid = _get_pid()
    if pid and _daemon_running(pid):
        state = _green("running")
        pid_str = _green(str(pid))
    else:
        state = _red("stopped")
        pid_str = _dim("none")

    print(_bold("AIOS-Core Status"))
    print(f"  Version  : {VERSION}")
    print(f"  State    : {state}")
    print(f"  PID      : {pid_str}")

    # Config summary
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text())
            print(f"  Tick     : {cfg.get('tick_ms', 200)}ms")
            print(f"  Apply OS : {_green('yes') if cfg.get('apply_os') else _yellow('no (observe only)')}")
            model = cfg.get('model_path', 'none')
            model_ok = Path(model).exists() if model else False
            print(f"  Model    : {_green(model) if model_ok else _red(str(model) + ' (missing)')}")
        except Exception:
            pass

    # Audit log summary
    if AUDIT_LOG.exists():
        lines = AUDIT_LOG.read_text().splitlines()
        events = [json.loads(l) for l in lines if l.strip()]
        rollbacks   = sum(1 for e in events if e.get("event") == "rollback")
        quarantines = sum(1 for e in events if e.get("event") == "quarantine_lifted")
        applies     = sum(1 for e in events if e.get("event") == "apply")
        print(f"\n  {_bold('Audit log')} ({len(events)} entries)")
        print(f"    Weight changes : {applies}")
        print(f"    Rollbacks      : {_yellow(str(rollbacks)) if rollbacks else '0'}")
        print(f"    Quarantines    : {_yellow(str(quarantines)) if quarantines else '0'}")
        if events:
            last = events[-1]
            age = time.time() - last.get("ts", 0)
            print(f"    Last event     : {last.get('event')} ({age:.0f}s ago, pid {last.get('pid')})")
    else:
        print(f"\n  {_dim('No audit log yet -- no weight changes recorded.')}")

    # cgroup check
    aios_cg = Path("/sys/fs/cgroup/aios")
    if aios_cg.exists():
        managed = list(aios_cg.glob("pid_*"))
        print(f"\n  {_bold('Cgroups')} : {_green(str(len(managed)))} PIDs under management")
        for d in managed[:5]:
            cw = (d / "cpu.weight").read_text().strip() if (d / "cpu.weight").exists() else "?"
            procs = (d / "cgroup.procs").read_text().strip() if (d / "cgroup.procs").exists() else ""
            print(f"    {_dim(d.name)}  cpu.weight={_cyan(cw)}  pids={procs}")
        if len(managed) > 5:
            print(f"    {_dim(f'... and {len(managed)-5} more')}")
    else:
        print(f"\n  {_dim('Cgroup /sys/fs/cgroup/aios not yet created.')}")


def cmd_top():
    """Live view of AI decisions -- refreshes every tick_ms."""
    try:
        import psutil
    except ImportError:
        print(_red("psutil not installed. Run: pip3 install psutil"))
        sys.exit(1)

    tick = 0.5   # refresh interval
    aios_cg = Path("/sys/fs/cgroup/aios")

    print(_bold("AIOS-Core Live  ") + _dim("(Ctrl+C to exit)\n"))

    try:
        while True:
            rows = []
            if aios_cg.exists():
                for d in sorted(aios_cg.glob("pid_*")):
                    try:
                        procs_text = (d / "cgroup.procs").read_text().strip()
                        if not procs_text:
                            continue
                        pid = int(procs_text.splitlines()[0])
                        cw_raw = (d / "cpu.weight").read_text().strip()
                        cw = int(cw_raw)
                        p = psutil.Process(pid)
                        name = p.name()[:18]
                        cpu = p.cpu_percent(interval=None)
                        status = p.status()
                        # Map cpu.weight back to [0,1] for bar display
                        cw_norm = min(cw / 200.0, 1.0)
                        rows.append((pid, name, cpu, cw, cw_norm, status))
                    except (OSError, ValueError, psutil.NoSuchProcess):
                        continue

            # Clear and redraw
            os.system("clear")
            print(_bold(f"AIOS-Core Live") +
                  f"  {_dim('tick every 200ms')}  "
                  f"managed={_cyan(str(len(rows)))}  "
                  f"{'  ' + _green('daemon: running') if (_get_pid() and _daemon_running(_get_pid())) else _red('daemon: stopped')}")
            print()

            if rows:
                header = (f"{'PID':>7}  {'Name':<18}  {'CPU%':>5}  "
                          f"{'cpu.weight':>10}  {'weight bar':<12}  {'status'}")
                print(_bold(header))
                print(_dim("-" * 70))
                for pid, name, cpu, cw, cw_norm, status in rows:
                    bar = _bar(cw_norm)
                    cw_str = _green(str(cw)) if cw > 100 else (_red(str(cw)) if cw < 100 else _dim(str(cw)))
                    print(f"{pid:>7}  {name:<18}  {cpu:>5.1f}  {cw_str:>10}  {bar}  {_dim(status)}")
            else:
                print(_dim("  No processes under active management yet."))
                print(_dim("  The daemon observes all processes; cgroups are created"))
                print(_dim("  as the AI starts issuing non-default weight decisions."))

            time.sleep(tick)

    except KeyboardInterrupt:
        print("\nExiting.")


def cmd_log(n: int = 20):
    if not AUDIT_LOG.exists():
        print(_dim("No audit log yet."))
        return
    lines = AUDIT_LOG.read_text().splitlines()[-n:]
    for line in lines:
        try:
            e = json.loads(line)
            ts = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
            ev = e["event"]
            pid = e["pid"]
            color = _green if ev == "apply" else (_red if ev == "rollback" else _yellow)
            extra = {k: v for k, v in e.items()
                     if k not in ("ts", "event", "pid", "tick")}
            print(f"{_dim(ts)}  {color(ev):<12}  pid={pid:<6}  {extra}")
        except Exception:
            print(line)


def cmd_stop():
    pid = _get_pid()
    if not pid or not _daemon_running(pid):
        print(_yellow("Daemon is not running."))
        return
    os.kill(pid, signal.SIGTERM)
    print(f"Sent SIGTERM to pid {pid}. Waiting...")
    for _ in range(10):
        time.sleep(0.5)
        if not _daemon_running(pid):
            print(_green("Daemon stopped."))
            return
    print(_red("Daemon did not stop in 5s. Try: kill -9 " + str(pid)))


def cmd_reload():
    pid = _get_pid()
    if not pid or not _daemon_running(pid):
        print(_red("Daemon is not running."))
        return
    os.kill(pid, signal.SIGHUP)
    print(_green(f"Sent SIGHUP to pid {pid}. Model weights will reload."))


# ── Entry point ───────────────────────────────────────────────────────────────

COMMANDS = {
    "status":  (cmd_status,  "Show daemon health and audit summary"),
    "top":     (cmd_top,     "Live view of AI scheduling decisions"),
    "log":     (cmd_log,     "Tail the audit log"),
    "stop":    (cmd_stop,    "Gracefully stop the daemon"),
    "reload":  (cmd_reload,  "Hot-reload model weights (no restart)"),
    "version": (cmd_version, "Print version"),
}

def usage():
    print(_bold("aios") + f" v{VERSION} -- AI process scheduler")
    print(f"\nUsage: aios <command>\n")
    for cmd, (_, desc) in COMMANDS.items():
        print(f"  {_cyan(cmd):<10} {desc}")
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        usage()
        sys.exit(0 if len(sys.argv) < 2 else 1)

    cmd = sys.argv[1]
    fn = COMMANDS[cmd][0]
    if cmd == "log" and len(sys.argv) > 2:
        fn(int(sys.argv[2]))
    else:
        fn()
