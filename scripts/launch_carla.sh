#!/usr/bin/env bash
# Launch a single headless CARLA server instance pinned to a specific GPU
# and RPC port. Parameterized (not hardcoded) because Phase 6 needs to
# launch several instances in parallel, each on its own GPU/port.
#
# Usage:
#   ./launch_carla.sh <gpu_id> <port> [carla_root] [quality_level]
#
# Env vars (used if the matching positional arg is omitted):
#   CARLA_ROOT          path to an extracted CARLA package (contains
#                       CarlaUE4.sh and PythonAPI/) - only used to locate
#                       logs / stay consistent with the rest of docs/setup.md
#   CARLA_QUALITY       rendering quality level (default: Low; fine for
#                       headless RL/eval since nothing is actually displayed)
#   CARLA_LAUNCH_MODE   "docker" (default) or "native"
#   CARLA_DOCKER_IMAGE  docker image to run in docker mode
#                       (default: carlasim/carla:0.9.11)
#
# Why docker mode is the default: as of this project's setup, CARLA's own S3
# tarball hosting for 0.9.11 is dead (see docs/setup.md §0/§4 for exactly how
# that was diagnosed) and the official carlasim/carla:0.9.11 Docker Hub image
# is the working substitute recommended by CARLA's own docs for headless
# Linux. "native" mode is kept for a host where a real CARLA_ROOT/CarlaUE4.sh
# install exists (e.g. the tarball path comes back, or a different node has
# it pre-installed).
#
# Examples:
#   ./launch_carla.sh 3 2000
#   CARLA_LAUNCH_MODE=native CARLA_ROOT=/data/savyo/carla-redteam/CARLA_0.9.11 ./launch_carla.sh 3 2000

set -euo pipefail

GPU_ID="${1:?Usage: $0 <gpu_id> <port> [carla_root] [quality_level]}"
PORT="${2:?Usage: $0 <gpu_id> <port> [carla_root] [quality_level]}"
CARLA_ROOT="${3:-${CARLA_ROOT:-}}"
QUALITY="${4:-${CARLA_QUALITY:-Low}}"
LAUNCH_MODE="${CARLA_LAUNCH_MODE:-docker}"
DOCKER_IMAGE="${CARLA_DOCKER_IMAGE:-carlasim/carla:0.9.11}"

LOG_DIR="${CARLA_LOG_DIR:-${CARLA_ROOT:-.}/../logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/carla_gpu${GPU_ID}_port${PORT}.log"

echo "Launching CARLA: mode=${LAUNCH_MODE} gpu=${GPU_ID} port=${PORT} quality=${QUALITY}"
echo "Log: ${LOG_FILE}"

if [[ "$LAUNCH_MODE" == "docker" ]]; then
  CONTAINER_NAME="carla_gpu${GPU_ID}_port${PORT}"
  # Persistent shader-cache volume (survives container recreation, e.g. a
  # relaunch after a crash) - UE4/the NVIDIA driver both cache compiled
  # Vulkan shaders under $HOME=/home/carla in the image; without this every
  # fresh container recompiles from scratch, which can be slow enough on
  # first load of a new map to trip client-side RPC timeouts. The "carla"
  # user in the image is uid 1000, unlikely to match the host account
  # running this script, hence the chmod.
  CACHE_DIR="${CARLA_LOG_DIR:-${CARLA_ROOT:-.}/../logs}/../cache/carla_home_gpu${GPU_ID}"
  mkdir -p "$CACHE_DIR"
  chmod 777 "$CACHE_DIR"
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus "device=${GPU_ID}" \
    --net=host \
    --shm-size=2g \
    -e SDL_VIDEODRIVER=offscreen \
    -v "$(cd "$CACHE_DIR" && pwd)":/home/carla/.config \
    "$DOCKER_IMAGE" \
    ./CarlaUE4.sh \
      -RenderOffScreen \
      -carla-rpc-port="$PORT" \
      -quality-level="$QUALITY" \
      -nosound \
      -nocrashreports \
    > "$LOG_FILE" 2>&1

  echo "Container: ${CONTAINER_NAME}"
  # Map load (UE4 level streaming, shader setup) can take well over 8s on
  # first run; poll instead of a single fixed sleep.
  sleep 20
  if ! docker ps --filter "name=${CONTAINER_NAME}" --filter "status=running" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; then
    echo "error: CARLA container exited immediately, check 'docker logs ${CONTAINER_NAME}'" >&2
    docker logs "$CONTAINER_NAME" >&2 || true
    exit 1
  fi
  docker logs "$CONTAINER_NAME" > "$LOG_FILE" 2>&1 || true

else
  CARLA_BIN="$CARLA_ROOT/CarlaUE4.sh"
  if [[ ! -x "$CARLA_BIN" ]]; then
    echo "error: CARLA binary not found or not executable at $CARLA_BIN" >&2
    exit 1
  fi

  CUDA_VISIBLE_DEVICES="$GPU_ID" "$CARLA_BIN" \
    -RenderOffScreen \
    -carla-rpc-port="$PORT" \
    -quality-level="$QUALITY" \
    -nosound \
    > "$LOG_FILE" 2>&1 &

  CARLA_PID=$!
  echo "CARLA PID: ${CARLA_PID}"

  sleep 5
  if ! kill -0 "$CARLA_PID" 2>/dev/null; then
    echo "error: CARLA process exited immediately, check ${LOG_FILE}" >&2
    exit 1
  fi
fi

echo "CARLA appears to be running. Verify with:"
echo "  python scripts/check_client_connection.py --host localhost --port ${PORT}"
