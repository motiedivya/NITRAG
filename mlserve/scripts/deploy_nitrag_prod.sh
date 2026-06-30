#!/usr/bin/env bash
# deploy_nitrag_prod.sh — Deploy NITRAG FastAPI server on mlcpu (prod).
# Run directly on mlcpu as root, or via: ssh root@10.9.0.36 'bash -s' < deploy_nitrag_prod.sh
#
# What it does:
#   1. Installs uv (fast Python package manager)
#   2. Clones / pulls repo to /opt/nitrag
#   3. Creates .venv with Python 3.12 and all dependencies
#   4. Creates /opt/nitrag/.env from template (if not already present)
#   5. Installs and starts nitrag-server.service
#   6. Health-checks the running server

set -euo pipefail

REPO_URL="https://github.com/motiedivya/NITRAG.git"
REPO_DIR="/opt/nitrag"
ENV_FILE="${REPO_DIR}/.env"
TEMPLATE_FILE="${REPO_DIR}/configs/env.prod.example"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 1. Install uv ─────────────────────────────────────────────────────────────
log "Checking uv..."
if ! command -v uv &>/dev/null; then
    log "Installing uv..."
    curl -fsSL https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    log "uv installed: $(uv --version)"
else
    log "uv already present: $(uv --version)"
fi
export PATH="$HOME/.local/bin:$PATH"

# ── 2. Clone / update repo ────────────────────────────────────────────────────
log "Syncing repo ${REPO_URL} → ${REPO_DIR}..."
if [ -d "${REPO_DIR}/.git" ]; then
    git -C "${REPO_DIR}" fetch --depth=1 origin main
    git -C "${REPO_DIR}" reset --hard origin/main
    log "Repo updated"
else
    git clone --depth=1 "${REPO_URL}" "${REPO_DIR}"
    log "Repo cloned"
fi

# ── 3. Python venv with uv ────────────────────────────────────────────────────
log "Setting up Python 3.12 venv with uv..."
cd "${REPO_DIR}"
uv python install 3.12 --quiet
uv venv --python 3.12 --quiet
uv pip install --quiet -e ".[dev]"
log "venv ready: $(${REPO_DIR}/.venv/bin/python --version)"

# ── 4. .env file ──────────────────────────────────────────────────────────────
if [ ! -f "${ENV_FILE}" ]; then
    log "Creating ${ENV_FILE} from template..."
    cp "${TEMPLATE_FILE}" "${ENV_FILE}"
    log ""
    log "  *** ACTION REQUIRED ***"
    log "  Edit ${ENV_FILE} and set OPENAI_API_KEY to your real key."
    log "  Then restart: systemctl restart nitrag-server"
    log ""
else
    log ".env already exists — not overwriting"
fi

# ── 5. Systemd service ────────────────────────────────────────────────────────
log "Installing nitrag-server.service..."
cp "${REPO_DIR}/mlserve/systemd/nitrag-server.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now nitrag-server
log "nitrag-server: $(systemctl is-active nitrag-server)"

# ── 6. Health check ───────────────────────────────────────────────────────────
log "Waiting for server to start..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/api/health &>/dev/null; then
        log "Server healthy: $(curl -s http://localhost:8000/api/health)"
        break
    fi
    if [ $i -eq 30 ]; then
        log "Server did not start in 30s — check: journalctl -u nitrag-server -n 50"
        exit 1
    fi
    sleep 2
done

log ""
log "Done. NITRAG prod is running at http://$(hostname -I | awk '{print $1}'):8000"
log "API docs: http://$(hostname -I | awk '{print $1}'):8000/docs"
log "Logs: journalctl -u nitrag-server -f"
