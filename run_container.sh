#!/usr/bin/env bash

set -euo pipefail

GPU_ID="${1:-0}"
MODE="${2:-bash}"
JUPYTER_PORT="${3:-8888}"
IMAGE_NAME="${IMAGE_NAME:-raywave_tracing:latest}"
CONTAINER_NAME="raywave_gpu${GPU_ID}"

COMMON_ARGS=(
    --gpus "device=${GPU_ID}"
    --name "${CONTAINER_NAME}"
    -v "$PWD:/app"
    -w /app
    -e MPLBACKEND=Agg
    -e MPLCONFIGDIR=/tmp/matplotlib
    -e PYTHONPATH=/app
)

if [[ "${MODE}" == "jupyter" ]]; then
    echo "Starting Jupyter Notebook on http://localhost:${JUPYTER_PORT}"
    docker run --rm -it \
        "${COMMON_ARGS[@]}" \
        -p "${JUPYTER_PORT}:8888" \
        "${IMAGE_NAME}" \
        jupyter notebook \
            --ip=0.0.0.0 \
            --port=8888 \
            --no-browser \
            --allow-root \
            --NotebookApp.token='' \
            --NotebookApp.password=''
else
    docker run --rm -it \
        "${COMMON_ARGS[@]}" \
        "${IMAGE_NAME}" \
        bash
fi
