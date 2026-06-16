#!/usr/bin/env bash
# AIOS-Core Installer
# Usage: curl -sSL https://raw.githubusercontent.com/frostfrazer/aios-core/master/install.sh | sudo bash
set -euo pipefail

REPO="https://github.com/frostfrazer/aios-core"
RAW="https://raw.githubusercontent.com/frostfrazer/aios-core/master"
INSTALL_DIR="/opt/aios-core"
CONFIG_DIR="/etc/aios"
LOG_DIR="/var/log/aios"
BIN="/usr/bin/aios-daemon"
SERVICE="/etc/systemd/system/aios.service"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[aios]${NC} $*"; }
warn()    { echo -e "${YELLOW}[aios]${NC} $*"; }
err()     { echo -e "${RED}[aios]${NC} $*" >&2; exit 1; }

# ── Checks ────────────────────────────────────────────────────────────────────

[[ $EUID -ne 0 ]] && err "Run as root: sudo bash install.sh"

command -v python3 >/dev/null 2>&1 || err "python3 not found. Install it first."
PY=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python $PY detected."

# Check cgroup v2
if ! mountpoint -q /sys/fs/cgroup; then
    warn "cgroup v2 not mounted. CPU weight tuning will be disabled."
fi

# ── Install deps ──────────────────────────────────────────────────────────────

info "Installing Python dependencies..."
python3 -m pip install numpy psutil --quiet --break-system-packages 2>/dev/null \
    || python3 -m pip install numpy psutil --quiet

# ── Install AIOS-Core ─────────────────────────────────────────────────────────

info "Installing AIOS-Core to $INSTALL_DIR..."
if command -v git >/dev/null 2>&1 && [[ ! -d "$INSTALL_DIR/.git" ]]; then
    git clone --depth=1 "$REPO" "$INSTALL_DIR" 2>/dev/null || true
elif [[ -d "$INSTALL_DIR/.git" ]]; then
    git -C "$INSTALL_DIR" pull --quiet
else
    # Fallback: download key files via curl
    mkdir -p "$INSTALL_DIR"/{models,orchestrator,telemetry,simulator,weights}
    for f in daemon.py models/neural_scheduler.py telemetry/collector.py \
              orchestrator/engine.py orchestrator/actuator.py \
              orchestrator/safety.py \
              models/__init__.py orchestrator/__init__.py telemetry/__init__.py; do
        curl -sSL "$RAW/$f" -o "$INSTALL_DIR/$f"
    done
fi

# ── Wrapper binary ────────────────────────────────────────────────────────────

cat > "$BIN" <<WRAPPER
#!/usr/bin/env bash
exec python3 $INSTALL_DIR/daemon.py "\$@"
WRAPPER
chmod +x "$BIN"

# ── Directories and config ────────────────────────────────────────────────────

mkdir -p "$CONFIG_DIR" "$LOG_DIR"
chmod 750 "$LOG_DIR"

if [[ ! -f "$CONFIG_DIR/config.json" ]]; then
    info "Writing default config to $CONFIG_DIR/config.json..."
    cat > "$CONFIG_DIR/config.json" <<CONFIG
{
  "tick_ms":    200,
  "apply_os":   true,
  "model_path": "/etc/aios/weights.json",
  "log_level":  "INFO",
  "pid_file":   "/run/aios/aios.pid",
  "audit_log":  "/var/log/aios/audit.jsonl"
}
CONFIG
else
    info "Config already exists at $CONFIG_DIR/config.json, skipping."
fi

# Copy trained model weights
if [[ -f "$INSTALL_DIR/weights/trained.json" ]]; then
    cp "$INSTALL_DIR/weights/trained.json" "$CONFIG_DIR/weights.json"
    info "Trained model weights installed."
else
    warn "No trained weights found -- daemon will start with random weights."
    warn "Run: python3 $INSTALL_DIR/models/train.py --out /etc/aios/weights.json"
fi

# ── Systemd service ───────────────────────────────────────────────────────────

info "Installing systemd service..."
curl -sSL "$RAW/aios.service" -o "$SERVICE" 2>/dev/null \
    || cp "$INSTALL_DIR/aios.service" "$SERVICE"

systemctl daemon-reload
systemctl enable aios
systemctl restart aios

sleep 2
if systemctl is-active --quiet aios; then
    info "AIOS-Core is running."
    info "  Status:  systemctl status aios"
    info "  Logs:    journalctl -u aios -f"
    info "  Audit:   tail -f /var/log/aios/audit.jsonl"
    info "  Config:  $CONFIG_DIR/config.json"
    info "  Reload:  systemctl kill -s HUP aios  (hot-reload model weights)"
    info "  Stop:    systemctl stop aios"
else
    err "Service failed to start. Check: journalctl -u aios -n 50"
fi
