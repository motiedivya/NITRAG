#!/usr/bin/env bash
# deploy_mlcpu.sh — Deploy NSQ infrastructure + CPU consumers on mlcpu (10.9.0.36)
# Run this directly on mlcpu as root, or via: ssh root@10.9.0.36 'bash -s' < deploy_mlcpu.sh
#
# What it does:
#   1. Installs NSQ binaries (nsqd, nsqlookupd, nsqadmin)
#   2. Clones / pulls the NITRAG repo to /opt/nitrag (read-only HTTPS)
#   3. Creates /opt/nitrag-venv with pynsq, fastembed, tiktoken
#   4. Installs systemd services for nsqlookupd, nsqd, nsqadmin, nitrag-embed, nitrag-chunk
#   5. Enables and starts all services

set -euo pipefail

REPO_URL="https://github.com/motiedivya/NITRAG.git"
REPO_DIR="/opt/nitrag"
VENV_DIR="/opt/nitrag-venv"
NSQ_VERSION="1.3.0"
NSQ_ARCH="linux-amd64"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 1. Install NSQ binaries ───────────────────────────────────────────────────
log "Installing NSQ ${NSQ_VERSION}..."
if ! command -v nsqd &>/dev/null; then
    cd /tmp
    curl -fsSL "https://github.com/nsqio/nsq/releases/download/v${NSQ_VERSION}/nsq-${NSQ_VERSION}.${NSQ_ARCH}.tar.gz" \
        -o nsq.tar.gz
    tar xzf nsq.tar.gz
    cp "nsq-${NSQ_VERSION}.${NSQ_ARCH}/bin/nsqd"       /usr/local/bin/
    cp "nsq-${NSQ_VERSION}.${NSQ_ARCH}/bin/nsqlookupd"  /usr/local/bin/
    cp "nsq-${NSQ_VERSION}.${NSQ_ARCH}/bin/nsqadmin"    /usr/local/bin/
    chmod +x /usr/local/bin/nsq*
    rm -rf /tmp/nsq.tar.gz "/tmp/nsq-${NSQ_VERSION}.${NSQ_ARCH}"
    log "NSQ installed: $(nsqd --version)"
else
    log "NSQ already installed: $(nsqd --version)"
fi

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

# ── 3. Python venv ────────────────────────────────────────────────────────────
log "Setting up Python venv at ${VENV_DIR}..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/pip" install --quiet --upgrade pip
"${VENV_DIR}/bin/pip" install --quiet \
    "pynsq>=0.9.1" \
    "fastembed>=0.4.0" \
    "tiktoken>=0.13.0" \
    "numpy>=1.24"
log "Python venv ready"

# ── 4. Systemd services ───────────────────────────────────────────────────────
SYSTEMD_SRC="${REPO_DIR}/mlserve/systemd"

for svc in nsqlookupd nsqd nsqadmin nitrag-embed nitrag-chunk; do
    if [ -f "${SYSTEMD_SRC}/${svc}.service" ]; then
        cp "${SYSTEMD_SRC}/${svc}.service" "/etc/systemd/system/${svc}.service"
        log "Installed ${svc}.service"
    fi
done

systemctl daemon-reload

# ── 5. Enable + start services ────────────────────────────────────────────────
for svc in nsqlookupd nsqd nsqadmin; do
    systemctl enable --now "${svc}"
    log "${svc}: $(systemctl is-active ${svc})"
done

# Give NSQ a moment to come up before starting consumers
sleep 2

for svc in nitrag-embed nitrag-chunk; do
    systemctl enable --now "${svc}"
    log "${svc}: $(systemctl is-active ${svc})"
done

log ""
log "Done. NSQ admin UI: http://$(hostname -I | awk '{print $1}'):4171"
log "Check logs: journalctl -u nitrag-embed -f"
