# AIOS-Core

AI-driven process scheduler for Linux. Uses a neural network to dynamically tune CPU and I/O cgroup weights per process, improving mixed-workload throughput by 33–48%.

## How it works

A lightweight feed-forward net (12→32→16→4) observes per-process telemetry every 200ms and outputs CPU/IO weight adjustments. A SafetyGuardian rolls back and quarantines any process whose reward degrades below threshold. Everything runs as a systemd daemon with zero user intervention.

## Requirements

- Linux with cgroup v2 (`/sys/fs/cgroup` mounted)
- Python 3.10+
- `numpy`, `psutil`
- Root (required for cgroup writes)

## Install (one-liner)

```bash
curl -sSL https://raw.githubusercontent.com/frostfrazer/aios-core/master/install.sh | sudo bash
```

## CLI

```bash
aios status      # daemon health, guardian stats, managed PIDs
aios top         # live per-process weight display
aios log         # tail the audit log
aios reload      # hot-reload model weights (SIGHUP)
aios stop        # graceful shutdown
aios version
```

## Config

`/etc/aios/config.json`

```json
{
  "tick_ms":    200,
  "apply_os":   true,
  "model_path": "/etc/aios/weights.json",
  "log_level":  "INFO",
  "pid_file":   "/run/aios/aios.pid",
  "audit_log":  "/var/log/aios/audit.jsonl"
}
```

## Logs

| Path | Contents |
|------|----------|
| `journalctl -u aios -f` | Daemon stdout/stderr, quarantine alerts |
| `/var/log/aios/audit.jsonl` | Per-PID weight changes, rollbacks, quarantines |

Logrotate installed at `/etc/logrotate.d/aios` — 50MB cap, 3 rotations, compressed.

## Safety

- **Rollback**: if a PID's mean reward drops below threshold, weights reset to default
- **Quarantine**: PID is locked out for 25 ticks before re-admission
- **Alert**: quarantine events print to stderr → captured by journald

## Benchmark

CI results on a 2-vCPU runner (mixed CPU + I/O workload):

| Workload | Baseline | AIOS | Improvement |
|----------|----------|------|-------------|
| Mixed workers | 100% | 133–148% | +33–48% |
| Net result | — | — | +0.31% CI |

## Project

Part of the **Frost** portfolio — tools for African developers and SMEs.  
Repo: [github.com/frostfrazer/aios-core](https://github.com/frostfrazer/aios-core)
