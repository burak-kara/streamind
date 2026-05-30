#!/usr/bin/env bash
#
# pipeline_prep.sh — shared launch-prep for the pipeline runners.
#
# Sourced (not executed) by tools/run_pipeline.sh and tools/compare_summarizers.sh
# so the two cannot drift. Single source of truth for the runtime environment and
# the local-weights preflight. Assumes the caller's CWD is the repo root.
#
# Why this exists: faster-whisper (CTranslate2) dlopen's libcublas.so.12, which
# pip installs under site-packages/nvidia/*/lib — NOT on the default loader path.
# A launcher that forgets to export it crashes ASR at warmup. Keeping the export
# here means every launcher inherits it.

# Echo the configured model_name for a summarizer/judge profile JSON. $1 = path.
model_path_of() {
  python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['configuration']['model_name'])" "$1"
}

# Echo the results subdir a run writes to for a profile JSON. $1 = path.
# Mirrors result_transmitter: basename(model_name) with ':' -> '-'. The pipeline
# always writes window_*.json under results/<this>/<window>/ regardless of run,
# so both launchers use it to locate (and pre-clean) that fixed output dir.
model_dir_of() {
  local mp; mp=$(model_path_of "$1") || return 1
  basename "${mp%/}" | tr ':' '-'
}

# Abort unless a CUDA GPU is visible. Returns nonzero (no exit) so the caller
# keeps its own error style.
require_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 && return 0
  echo "ERROR: nvidia-smi not found. CUDA pipeline requires a GPU host." >&2
  return 1
}

# Verify a profile's weights are present locally. $1 = profile JSON, $2 = label
# for the message (e.g. summarizer/judge). Returns nonzero on failure (no exit).
require_model() {
  local pf="$1" kind="${2:-model}" mp
  mp=$(model_path_of "$pf") || { echo "ERROR: cannot read model_name from $pf" >&2; return 1; }
  if [ ! -f "$mp/config.json" ]; then
    echo "ERROR: $kind weights missing: $mp" >&2
    echo "Run: ./tools/fetch_models.sh <hf_id> $(basename "$mp")" >&2
    return 1
  fi
  return 0
}

# Export the runtime env every in-process vLLM + faster-whisper launch needs:
#   - VLLM_USE_FLASHINFER_SAMPLER=0 : skip the nvcc-JIT sampler kernel (the box
#     has the CUDA runtime but no toolkit); the native top-k/p path is ~equal
#     latency and avoids a multi-GB toolkit dependency + 10-30 s first-window JIT.
#   - LD_LIBRARY_PATH += pip nvidia cublas/cuda_runtime lib dir : CTranslate2
#     (faster-whisper backend) links libcublas.so.12 from site-packages, off the
#     default loader path. Without this, ASR fails to load at warmup.
prep_runtime_env() {
  export VLLM_USE_FLASHINFER_SAMPLER=0
  local cublas_dir
  cublas_dir=$(uv run python -c "
import importlib.util, pathlib
for pkg in ('nvidia.cublas', 'nvidia.cuda_runtime'):
    spec = importlib.util.find_spec(pkg)
    if spec and spec.submodule_search_locations:
        lib = pathlib.Path(spec.submodule_search_locations[0]) / 'lib'
        if lib.is_dir():
            print(lib); break
" 2>/dev/null)
  if [ -n "${cublas_dir:-}" ] && [ -d "$cublas_dir" ]; then
    export LD_LIBRARY_PATH="${cublas_dir}:${LD_LIBRARY_PATH:-}"
  fi
}
