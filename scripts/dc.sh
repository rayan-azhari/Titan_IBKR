#!/bin/bash
# dc.sh -- docker compose wrapper that bakes in --env-file .env.docker.
#
# V3.7 / L68: every compose operation that recreates or starts containers
# MUST source .env.docker, otherwise IB Gateway loses TWS credentials and
# hangs at "Setting password" (root cause of Phase 2 cutover initial
# failure, 2026-05-17). The compose CLI emits a warning but it's easy to
# miss in the cascade of Recreate/Started output.
#
# This wrapper makes the convention foolproof:
#   ./scripts/dc.sh up -d --force-recreate titan-portfolio
#   ./scripts/dc.sh restart ib-gateway
#   ./scripts/dc.sh build titan-portfolio
#   ./scripts/dc.sh down
#
# Usage from the project root.

set -euo pipefail

# Resolve project root so the script works from any directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env.docker"

if [ ! -f "${ENV_FILE}" ]; then
  echo "ERROR: ${ENV_FILE} not found. Required for V3.7 compose ops (L68)." >&2
  echo "  Copy from .env.docker.example and fill in IBKR credentials." >&2
  exit 1
fi

# Pre-flight: verify the critical env vars are present (not just the file).
required_keys=("TWS_USERID" "TWS_PASSWORD" "IBKR_ACCOUNT_ID")
missing=()
for key in "${required_keys[@]}"; do
  if ! grep -qE "^${key}=." "${ENV_FILE}"; then
    missing+=("${key}")
  fi
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "ERROR: .env.docker is missing required keys: ${missing[*]}" >&2
  echo "  Without these, IB Gateway will hang at 'Setting password' (L68)." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
exec docker compose --env-file "${ENV_FILE}" "$@"
