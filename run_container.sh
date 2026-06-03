#!/usr/bin/env bash

set -e

GPU_ID=${1:-0}
IMAGE_NAME="chengjz_torch:latest"
CONTAINER_NAME="raywave_gpu${GPU_ID}"

docker run --rm -it \
    --gpus "\"device=${GPU_ID}\"" \
    --name "${CONTAINER_NAME}" \
    -v "$PWD:/app" \
    -w /app \
    -e MPLBACKEND=Agg \
    -e PYTHONPATH=/app \
    "${IMAGE_NAME}" \
    bash