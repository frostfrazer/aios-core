# AIOS-Core Development Log

## Session 1 — Core pipeline + cgroups actuator

### What was built
- `models/neural_scheduler.py` — numpy feed-forward net (12->32->16->4),
  online RL-lite weight update, save/load.
- `telemetry/collector.py` — real OS telemetry via psutil, 12-dim
  normalised feature vectors, cached Process objects.
- `orchestrator/engine.py` — closed-loop control (200ms default tick),
  reward computation, online learning toggle.
- `orchestrator/actuator.py` — cgroups v2 actuation (cpu.weight,
  io.weight where BFQ available), per-PID cgroup lifecycle.
- `orchestrator/safety.py` — SafetyGuardian: rate-limited weight changes,
  automatic rollback + quarantine on sustained negative reward, JSONL
  audit log.
- `models/train.py` — offline trainer on synthetic labeled dataset
  (4 process classes: CPU-bound, IO-bound, mixed, idle). Converged
  0.37 -> 0.0135 MSE loss. Sanity checks confirm correct learned policy:
  CPU-bound -> nice_delta<0, cpu_weight>0.5; IO-bound -> nice_delta>0,
  io_weight>0.5.
- `simulator/workload.py` + `simulator/benchmark.py` — synthetic
  workload generator (cpu/io/mixed/idle) and A/B benchmark harness.
- `.github/workflows/benchmark.yml` — CI benchmark on isolated
  2-vCPU GitHub Actions runner.

### Key findings

**1. Trained model learns correct scheduling policy.** Verified via
sanity checks on synthetic CPU-bound and IO-bound feature vectors.

**2. Desktop/WSL benchmarking is too noisy for <10% effect sizes.**
Identical-config baseline runs varied 30-100% run-to-run due to
background load (browser, OS processes). GitHub Actions CI (isolated
2-vCPU runner) is the trustworthy measurement environment.

**3. CI result (first run, nice()-based actuation, pre-cgroups):**
Net throughput change -0.34% (noise-level), but per-worker breakdown
showed the AI redistributing priority from CPU-bound workers (-1.4% to
-3.4%) to IO-bound workers (+30% to +38%) and mixed workers (+8%).
The model is making real, directionally-correct decisions even when
net throughput is flat.

**4. CRITICAL BUG (found and fixed): redundant cgroup.procs writes.**
`CgroupActuator._ensure_pid_cgroup()` was re-writing `cgroup.procs` for
every watched PID on every tick, even when the PID was already resident
in its cgroup. At 200ms ticks x 16 PIDs, this is ~80 migrations/sec.
Measured impact on an 8-core WSL oversubscription benchmark (16 procs):
**-19.2% throughput** with CPU utilisation unchanged (98.7% both
trials) -- i.e. pure waste, likely cgroup_mutex contention and cache
locality disruption for tight CPU loops, NOT resource starvation.

Fix: only write `cgroup.procs` on first migration (track via
`_managed_pids`); subsequent ticks only update `cpu.weight`/`io.weight`.
Result: -19.2% -> -2.4% (within this run's own baseline noise, which
swung 17505-34111 ops/s across identical configs).

**Lesson for the actuator design going forward:** any per-tick
filesystem write to a cgroup/proc control file must be audited for
whether it's actually idempotent at the KERNEL level, not just at the
Python level. "Harmless to call again" and "free to call again" are
very different things in cgroup v2.

### Validated safety mechanisms (smoke-tested on WSL Ubuntu 26.04)
- Per-PID cgroup creation, cpu.weight writes, cgroup.procs membership: OK
- Rate-limited convergence (max_delta=0.15/tick): confirmed, 0.5 -> 0.416
  over ~6 ticks as expected
- Full cleanup on stop(): cgroup directories removed, PIDs returned to
  system root cgroup
- Audit log: JSONL, one line per weight change / rollback / quarantine event
- Rollback/quarantine paths implemented but not yet triggered in testing
  (would need a workload that reliably produces sustained negative reward)

### Open items for next session
- Trigger and validate the rollback/quarantine path with an adversarial
  workload (e.g. a process whose performance genuinely degrades under
  AI control)
- io.weight is unavailable without BFQ (common on cloud VMs too) --
  consider io.max throttling as a secondary lever, with extra caution
  (hard limits are riskier than weight redistribution)
- Package as an installable daemon (systemd unit) for the "Frost
  AutoTune" product direction
- Broader workload validation beyond the 3 synthetic types
