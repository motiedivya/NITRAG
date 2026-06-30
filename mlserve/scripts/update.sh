#!/usr/bin/env bash
# update.sh — Pull latest code and restart consumers on the current server.
# Run this on each server after pushing changes to GitHub.
#
# Usage: ssh root@<server> 'bash /opt/nitrag/mlserve/scripts/update.sh'
#   Or add as a deploy hook in your CI.

set -euo pipefail

REPO_DIR="/opt/nitrag"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Pulling latest code..."
git -C "${REPO_DIR}" fetch --depth=1 origin main
git -C "${REPO_DIR}" reset --hard origin/main

# Detect which consumers live on this host and restart them
HOSTNAME_SHORT="$(hostname -s)"

declare -A SERVICE_HOSTS=(
    [nitrag-embed]="debian12"
    [nitrag-chunk]="debian12"
    [nitrag-ocr]="nit-rtx-6000"
    [nitrag-llm]="nit-rtx-6000"
)

RESTARTED=0
for svc in "${!SERVICE_HOSTS[@]}"; do
    if systemctl list-unit-files "${svc}.service" &>/dev/null && \
       systemctl is-enabled "${svc}" &>/dev/null; then
        systemctl restart "${svc}"
        log "Restarted ${svc}: $(systemctl is-active ${svc})"
        RESTARTED=$((RESTARTED+1))
    fi
done

if [ $RESTARTED -eq 0 ]; then
    log "No managed services found on this host (${HOSTNAME_SHORT})"
fi

log "Update complete."
