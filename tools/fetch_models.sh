#!/usr/bin/env bash
#
# Dev helper: download a HuggingFace model into ./models/<local_name>.
# Pipeline runtime never calls this — model weights ship with the
# submission (Dockerfile invokes the same script at build time).
#
# Usage:
#   ./tools/fetch_models.sh <hf_id> <local_name>
#
# Example:
#   ./tools/fetch_models.sh Qwen/Qwen3-8B qwen3-8b
#
# Idempotent: re-running with an existing populated dir is a no-op.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <hf_id> <local_name>" >&2
  echo "Example: $0 Qwen/Qwen3-8B qwen3-8b" >&2
  exit 1
fi

HF_ID="$1"
LOCAL_NAME="$2"
TARGET_DIR="./models/${LOCAL_NAME}"

if [[ -f "${TARGET_DIR}/config.json" ]]; then
  echo "Model already present at ${TARGET_DIR} (config.json found); skipping download." >&2
  exit 0
fi

mkdir -p "${TARGET_DIR}"

echo "Downloading ${HF_ID} -> ${TARGET_DIR}" >&2
uv run --extra dev huggingface-cli download \
  "${HF_ID}" \
  --local-dir "${TARGET_DIR}"

if [[ ! -f "${TARGET_DIR}/config.json" ]]; then
  echo "ERROR: download finished but ${TARGET_DIR}/config.json is missing." >&2
  echo "Inspect ${TARGET_DIR} and retry." >&2
  exit 2
fi

echo "Done. Pipeline configs should reference: ${TARGET_DIR}" >&2
