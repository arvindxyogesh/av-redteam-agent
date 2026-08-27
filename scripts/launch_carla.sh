#!/usr/bin/env bash
# Launch a single headless CARLA server instance pinned to a specific GPU
# and RPC port. Parameterized (not hardcoded) because Phase 6 needs to
# launch several instances in parallel, each on its own GPU/port.
#
# Usage:
#   ./launch_carla.sh <gpu_id> <port> [carla_root] [quality_level]
#
# Env vars (used if the matching positional arg is omitted):
#   CARLA_ROOT      path to an extracted CARLA package (contains CarlaUE4.sh)
#   CARLA_QUALITY   rendering quality level (default: Low; fine for headless
#                   RL/eval since nothing is actually displayed)
#
# Example:
#   ./launch_carla.sh 1 2000
#   CARLA_ROOT=/data/savyo/carla-redteam/CARLA_0.9.11 ./launch_carla.sh 0 3000

set -euo pipefail

GPU_ID="${1:?Usage: $0 <gpu_id> <port> [carla_root] [quality_level]}"
PORT="${2:?Usage: $0 <gpu_id> <port> [carla_root] [quality_level]}"
CARLA_ROOT="${3:-${CARLA_ROOT:?Set CARLA_ROOT or pass carla_root as the 3rd argument}}"
QUALITY="${4:-${CARLA_QUALITY:-Low}}"

CARLA_BIN="$CARLA_ROOT/CarlaUE4.sh"
if [[ ! -x "$CARLA_BIN" ]]; then
  echo "error: CARLA binary not found or not executable at $CARLA_BIN" >&2
  exit 1
fi

LOG_DIR="${CARLA_LOG_DIR:-$(dirname "$CARLA_ROOT")/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/carla_gpu${GPU_ID}_port${PORT}.log"

echo "Launching CARLA: gpu=${GPU_ID} port=${PORT} quality=${QUALITY} root=${CARLA_ROOT}"
echo "Log: ${LOG_FILE}"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$CARLA_BIN" \
  -RenderOffScreen \
  -carla-rpc-port="$PORT" \
  -quality-level="$QUALITY" \
  -nosound \
  > "$LOG_FILE" 2>&1 &

CARLA_PID=$!
echo "CARLA PID: ${CARLA_PID}"

# Give the server a moment to come up and do a basic liveness check.
sleep 5
if ! kill -0 "$CARLA_PID" 2>/dev/null; then
  echo "error: CARLA process exited immediately, check ${LOG_FILE}" >&2
  exit 1
fi

echo "CARLA appears to be running. Verify with:"
echo "  python scripts/check_client_connection.py --host localhost --port ${PORT}"
