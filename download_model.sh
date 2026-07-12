#!/usr/bin/env bash
# Downloads the bundled local model at Docker build time (needs internet on
# the build machine — Hugging Face is NOT reachable at grading runtime, so
# this must happen during `docker build`, not at container startup).
set -euo pipefail

MODEL_DIR="/app/models"
MODEL_FILE="qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/${MODEL_FILE}"

mkdir -p "${MODEL_DIR}"

if [ -f "${MODEL_DIR}/${MODEL_FILE}" ]; then
  echo "Model already present, skipping download."
  exit 0
fi

echo "Downloading ${MODEL_FILE} ..."
curl -L --fail --retry 3 -o "${MODEL_DIR}/${MODEL_FILE}" "${MODEL_URL}"
echo "Download complete: $(du -h "${MODEL_DIR}/${MODEL_FILE}" | cut -f1)"
