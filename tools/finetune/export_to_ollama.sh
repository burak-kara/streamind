#!/usr/bin/env bash
# Convert a fine-tuned summarizer to a GGUF + register it as an Ollama model.
#
# Pipeline:
#   1. Merge LoRA adapter → FP16 HF weights (merge_lora.py)
#   2. Clone llama.cpp on first run, install its Python deps
#   3. convert_hf_to_gguf.py → unquantized GGUF
#   4. llama-quantize → Q4_K_M GGUF (fits on the RTX 2070 with headroom)
#   5. ollama create <tag> -f Modelfile
#
# Usage:
#   tools/finetune/export_to_ollama.sh <run-dir> <ollama-tag> [quant]
# Example:
#   tools/finetune/export_to_ollama.sh \
#       tools/finetune/runs/qwen2.5-3b-rev16-r16 \
#       streamind-summarizer:rev16-r16 \
#       Q4_K_M

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <run-dir> <ollama-tag> [quant=Q4_K_M]" >&2
  exit 1
fi

RUN_DIR="$1"
OLLAMA_TAG="$2"
QUANT="${3:-Q4_K_M}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
LLAMA_CPP_DIR="$REPO_ROOT/tools/finetune/llama.cpp"
MERGED_DIR="$RUN_DIR/merged"
GGUF_RAW="$RUN_DIR/model.f16.gguf"
GGUF_QUANT="$RUN_DIR/model.${QUANT}.gguf"
MODELFILE="$RUN_DIR/Modelfile"

if [ ! -d "$RUN_DIR/adapter" ]; then
  echo "No adapter/ in $RUN_DIR — did finetune_summarizer.py finish?" >&2
  exit 1
fi

# Step 1: merge LoRA into base if not already done.
if [ ! -f "$MERGED_DIR/config.json" ]; then
  echo "[merge] $RUN_DIR/adapter → $MERGED_DIR"
  uv run python "$REPO_ROOT/tools/finetune/merge_lora.py" \
    --adapter-dir "$RUN_DIR" \
    --output-dir "$MERGED_DIR" \
    --device cpu
else
  echo "[merge] reusing existing $MERGED_DIR"
fi

# Step 2: ensure llama.cpp is available for the conversion + quantize binaries.
if [ ! -d "$LLAMA_CPP_DIR" ]; then
  echo "[llama.cpp] cloning"
  git clone --depth=1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
fi

if [ ! -f "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]; then
  echo "[llama.cpp] building llama-quantize"
  cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
  cmake --build "$LLAMA_CPP_DIR/build" --target llama-quantize -j
fi

# convert_hf_to_gguf.py needs a few python deps that aren't in our finetune extra.
# Install them into the same uv env on demand; cheap if they're already installed.
# `unsafe-best-match` lets uv look at PyPI for these deps even though our
# project pins torch to the pytorch-cu124 index — otherwise transformers gets
# resolved from the wrong index and the version conversion needs isn't found.
echo "[deps] ensuring llama.cpp Python deps"
uv pip install --quiet --index-strategy unsafe-best-match \
    -r "$LLAMA_CPP_DIR/requirements/requirements-convert_hf_to_gguf.txt"

# Step 3: HF → GGUF (FP16).
if [ ! -f "$GGUF_RAW" ]; then
  echo "[convert] $MERGED_DIR → $GGUF_RAW"
  uv run python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
    "$MERGED_DIR" --outfile "$GGUF_RAW" --outtype f16
else
  echo "[convert] reusing $GGUF_RAW"
fi

# Step 4: quantize.
if [ ! -f "$GGUF_QUANT" ]; then
  echo "[quantize] $QUANT → $GGUF_QUANT"
  "$LLAMA_CPP_DIR/build/bin/llama-quantize" "$GGUF_RAW" "$GGUF_QUANT" "$QUANT"
else
  echo "[quantize] reusing $GGUF_QUANT"
fi

# Step 5: write Modelfile and register with Ollama.
sed "s|{{GGUF_PATH}}|$GGUF_QUANT|" "$REPO_ROOT/tools/finetune/modelfile.template" > "$MODELFILE"
echo "[ollama] creating $OLLAMA_TAG from $MODELFILE"
ollama create "$OLLAMA_TAG" -f "$MODELFILE"

echo ""
echo "Done. Run with:"
echo "  ollama run $OLLAMA_TAG 'hello'"
echo "Or wire into the pipeline by setting model_name to '$OLLAMA_TAG' in a"
echo "summarizer profile under pipelines/summarizer/."
