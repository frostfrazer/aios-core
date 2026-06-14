"""Smoke test for cgroup actuator + safety guardian integration."""
import sys, time, subprocess, os
sys.path.insert(0, '.')

from orchestrator.engine import AIOrchestrator

print("--- Spawning a test CPU-burn worker ---")
proc = subprocess.Popen([sys.executable, "simulator/workload.py", "cpu", "8"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
pid = proc.pid
print(f"Worker PID: {pid}")

print("\n--- Starting orchestrator with apply_os=True (cgroups) ---")
orc = AIOrchestrator(tick_ms=200, apply_os=True, watch_pids=[pid])
print(f"Actuator available: {orc.actuator.available if orc.actuator else 'N/A (non-linux)'}")
print(f"io.weight available: {orc.actuator.io_weight_available if orc.actuator else 'N/A'}")

orc.start()
time.sleep(5)

# Check cgroup state for this pid
if orc.actuator and orc.actuator.available:
    cg_path = orc.actuator._pid_dir(pid)
    print(f"\n--- Cgroup state for pid {pid} ---")
    print(f"cgroup dir exists: {cg_path.exists()}")
    if cg_path.exists():
        cpu_w = (cg_path / "cpu.weight").read_text().strip()
        print(f"cpu.weight: {cpu_w}")
        procs = (cg_path / "cgroup.procs").read_text().strip()
        print(f"cgroup.procs: {procs}")

orc.stop()

print(f"\n--- After stop ---")
print(f"Total ticks: {orc.tick_count}")
print(f"Guardian stats: {orc.guardian.stats()}")
if orc.actuator:
    cg_path = orc.actuator._pid_dir(pid)
    print(f"cgroup dir still exists after cleanup: {cg_path.exists()}")

# Read audit log
print("\n--- Audit log (last 10 lines) ---")
try:
    with open("logs/safety_audit.jsonl") as f:
        lines = f.readlines()
        for line in lines[-10:]:
            print(line.strip())
    print(f"Total audit entries: {len(lines)}")
except FileNotFoundError:
    print("(no audit log written)")

stdout, _ = proc.communicate(timeout=10)
print(f"\n--- Worker output ---\n{stdout.strip()}")
print("\nDONE")
