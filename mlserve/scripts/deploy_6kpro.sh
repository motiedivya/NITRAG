#!/usr/bin/env bash
# deploy_6kpro.sh — Deploy GPU consumers on 6kpro (172.16.5.100)
# Run directly on 6kpro as root, or via: ssh root@172.16.5.100 'bash -s' < deploy_6kpro.sh
#
# What it does:
#   1. Clones / pulls the NITRAG repo to /opt/nitrag (read-only HTTPS)
#   2. Verifies /opt/paddle-ocr venv has required packages (pynsq + openai already present)
#   3. Installs systemd services for nitrag-ocr and nitrag-llm
#   4. Enables and starts both services

set -euo pipefail

REPO_URL="https://github.com/motiedivya/NITRAG.git"
REPO_DIR="/opt/nitrag"
PADDLE_PYTHON="/opt/paddle-ocr/bin/python"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 1. Clone / update repo ────────────────────────────────────────────────────
log "Syncing repo ${REPO_URL} → ${REPO_DIR}..."
if [ -d "${REPO_DIR}/.git" ]; then
    git -C "${REPO_DIR}" fetch --depth=1 origin main
    git -C "${REPO_DIR}" reset --hard origin/main
    log "Repo updated"
else
    git clone --depth=1 "${REPO_URL}" "${REPO_DIR}"
    log "Repo cloned"
fi

# ── 2. Verify paddle-ocr venv has required packages ──────────────────────────
log "Checking paddle-ocr venv packages..."
MISSING=0
for pkg in pynsq openai paddleocr; do
    if ! "${PADDLE_PYTHON}" -c "import ${pkg%%>=*}" &>/dev/null; then
        log "  MISSING: ${pkg}"
        MISSING=1
    else
        VER=$("${PADDLE_PYTHON}" -c "import importlib.metadata; print(importlib.metadata.version('${pkg}'))" 2>/dev/null || echo "?")
        log "  OK: ${pkg}==${VER}"
    fi
done
if [ $MISSING -eq 1 ]; then
    log "Installing missing packages into paddle-ocr venv..."
    /opt/paddle-ocr/bin/pip install --quiet pynsq openai
fi

# ── 3. Systemd services ───────────────────────────────────────────────────────
SYSTEMD_SRC="${REPO_DIR}/mlserve/systemd"

for svc in nitrag-ocr nitrag-llm; do
    if [ -f "${SYSTEMD_SRC}/${svc}.service" ]; then
        cp "${SYSTEMD_SRC}/${svc}.service" "/etc/systemd/system/${svc}.service"
        log "Installed ${svc}.service"
    else
        log "WARNING: ${SYSTEMD_SRC}/${svc}.service not found"
    fi
done

systemctl daemon-reload

# ── 4. Enable + start services ────────────────────────────────────────────────
for svc in nitrag-ocr nitrag-llm; do
    systemctl enable --now "${svc}"
    log "${svc}: $(systemctl is-active ${svc})"
done

log ""
log "Done."
log "Check OCR logs : journalctl -u nitrag-ocr -f"
log "Check LLM logs : journalctl -u nitrag-llm -f"
log "vLLM still on  : http://localhost:8000/v1/models"
