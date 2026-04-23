# Cloud GPU Development & Submission Readiness Plan

> Status: Draft — 2026-04-23
> Depends on: [submission-readiness-remaining.md](submission-readiness-remaining.md) (hard blockers #2, #3, #4)

## Problem Statement

The submission evaluation runs on a single **RTX Pro 4500** (24 GB VRAM, Ada).
Local development uses Apple Silicon + MLX, which **cannot run on NVIDIA hardware**.
The existing `Dockerfile`, `docker-compose.yml`, and `docker/entrypoint-pipeline.sh`
have never been built or tested end-to-end (blocker #2/#3 in submission-readiness).
The CUDA `proc_time` is unknown (measurement gap #4).

This plan covers:
1. Selecting a cloud GPU provider to close those blockers.
2. Setting up an efficient remote development workflow.
3. Exploring larger models on CUDA hardware (the 24 GB VRAM budget allows much more than 4B).
4. Fine-tuning open-source models on the provided `rev16` and `ietf` datasets.
5. Maintaining a dual-track inference backend strategy (Ollama vs. native vLLM).
6. Bringing the existing Dockerfile and compose setup to a verified, submission-ready state.

---

## 1. Cloud GPU Providers

### Selection Criteria
- **~24 GB VRAM** to match RTX Pro 4500 (Ada)
- **NVIDIA drivers + Docker with `nvidia-container-toolkit`** pre-installed
- **SSH access** for VS Code Remote
- **Hourly billing** (budget-friendly for burst testing)

### Recommended Providers

| Provider | GPU Options (24 GB) | Approx. Cost/hr | Notes |
| :--- | :--- | :--- | :--- |
| **RunPod** (top pick) | RTX A5000, RTX 4090, L4 | $0.20 – $0.50 | PyTorch/CUDA templates come with Docker + nvidia-toolkit. Easiest SSH setup. |
| **Vast.ai** | RTX 3090, RTX A5000, A10 | $0.15 – $0.40 | Community-hosted marketplace. Cheapest option. Hardware quality varies. |
| **Lambda Labs** | A10, A6000 | $0.50 – $0.75 | Reliable but availability can be limited. |
| **AWS** | `g5.xlarge` (A10G, 24 GB) | ~$1.00 | Enterprise-grade. Higher cost, more setup. |
| **GCP** | `g2-standard-4` (L4, 24 GB) | ~$0.70 | Good if you already have GCP credits. |

> **For fine-tuning** (Section 4), you may need a higher-VRAM instance temporarily.
> Consider an A100 (40/80 GB) or H100 on RunPod/Lambda for the training phase
> (~$1.50–$3.00/hr). Inference testing should remain on 24 GB instances to match
> the submission environment.

**Recommendation**: Start with **RunPod** using a standard PyTorch template and an RTX A5000 or RTX 4090 instance. These come pre-configured with CUDA drivers, Docker, and `nvidia-container-toolkit`, eliminating all driver setup.

---

## 2. Remote Development Workflow

### Initial Instance Setup

```bash
# 1. SSH into the cloud instance (credentials from provider dashboard)
ssh root@<instance-ip> -p <port>

# 2. Clone the repository
git clone <your-repo-url> streamind && cd streamind

# 3. Install uv (if not in the template)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 4. Create Python environment (no --extra mlx on Linux/CUDA)
uv sync --extra dev

# 5. Verify GPU is visible
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Editor Integration

Use **VS Code Remote - SSH** to get a local editing experience while executing on the cloud GPU:
1. Install the `Remote - SSH` extension in VS Code.
2. Add the cloud instance as an SSH host (`Cmd+Shift+P` → "Remote-SSH: Add New SSH Host").
3. Connect — VS Code opens a full workspace on the remote machine.
4. Terminal commands in VS Code run directly on the GPU instance.

### Audio Testing (Remote)

Since WebRTC from a local browser to a remote Janus instance requires HTTPS and port exposure, use the project's built-in audio injection tool instead:

```bash
# Terminal 1: Start the pipeline (runs on remote GPU)
./tools/run_pipeline.sh --window 30 -s ollama-qwen3.5-4b

# Terminal 2: Inject test audio through Janus
uv run python tools/send_audio.py tests/fixtures/youtube_15min.wav
```

This exactly matches the evaluation flow: Janus receives audio and forwards RTP to the pipeline on port 8888.

---

## 3. Model Size Exploration

### Why Larger Models Now Make Sense

On Apple Silicon with MLX, you are constrained by shared CPU/GPU memory and
the overhead of running everything in a single Python process. On the RTX Pro
4500 with 24 GB of dedicated VRAM, the budget opens up dramatically.

The scoring formula **heavily rewards quality over speed**:
- `B_i` (quality): max **25 points** — 5 Likert criteria × 5 each
- `K_i` (keywords): max **6 points**
- `L_i` (latency): max **~6 points** — and only if `B_i ≥ 10`

A model that scores `B_i = 22` at `proc_time = 4s` (`L_i ≈ 1.35`) yields
`C_i ≈ 29.35`. A smaller model scoring `B_i = 16` at `proc_time = 1s`
(`L_i ≈ 6.1`) yields `C_i ≈ 28.1`. **Quality wins.**

### VRAM Budget Analysis

| Component | VRAM (approx.) |
| :--- | :--- |
| Whisper `small.en` (faster-whisper, FP16) | ~1 GB |
| CUDA + framework overhead | ~2 GB |
| **Available for LLM** | **~21 GB** |

| Model | Quantization | VRAM (model + KV-cache) | Fits? |
| :--- | :--- | :--- | :--- |
| Qwen3.5-4B | 4-bit (Q4_K_M) | ~3 GB | ✅ Very comfortable |
| Qwen3.5-9B | 4-bit (Q4_K_M) | ~6 GB | ✅ Comfortable |
| Qwen3.5-9B | 8-bit (Q8_0) | ~10 GB | ✅ Fits well |
| Qwen3.5-14B | 4-bit (Q4_K_M) | ~9 GB | ✅ Fits well |
| Qwen3.5-14B | 8-bit (Q8_0) | ~16 GB | ⚠️ Tight but possible |
| Qwen3.5-32B | 4-bit (Q4_K_M) | ~19 GB | ⚠️ Very tight |

### Recommended Model Ladder

Benchmark these on the cloud GPU in ascending order. Stop when `proc_time`
exceeds 5s (at which point `L_i < 0.8`, making further size increases
unprofitable):

1. **Qwen3.5-4B Q4** — current baseline, expected `proc_time` ~1–2s
2. **Qwen3.5-9B Q4** — likely sweet spot, `proc_time` ~2–3s
3. **Qwen3.5-14B Q4** — if quality is still climbing, `proc_time` ~3–5s
4. **Qwen3.5-9B Q8** — same parameters as #2 but higher precision = better quality at same speed class

### Pipeline Profiles to Create

```bash
# New Ollama profiles needed in pipelines/summarizer/:
pipelines/summarizer/ollama-qwen3.5-9b.json     # already exists
pipelines/summarizer/ollama-qwen3.5-14b.json     # new
pipelines/summarizer/ollama-qwen3.5-9b-q8.json   # new (higher quant)
```

For Ollama, pulling larger models is trivial:
```bash
ollama pull qwen3.5:9b
ollama pull qwen3.5:14b
```

For vLLM, just change the `model` parameter in the node configuration.

---

## 4. Fine-Tuning Strategy

### Why Fine-Tune?

The current pipeline uses a generic base model with a prompt template. A model
fine-tuned specifically for meeting/podcast summarization could:
- Produce more relevant, concise summaries → higher `B_i`
- Reliably output exactly 3 domain-specific keywords → higher `K_i`
- Follow the JSON output format consistently → fewer parse failures
- Generate fewer tokens (learned brevity) → lower `proc_time` → higher `L_i`

A fine-tuned 4B model could potentially outperform a generic 14B model on this
specific task while being 3× faster.

### Available Training Data

The project includes two datasets:

| Dataset | Files | Content | Has Transcripts? | Has Summaries? |
| :--- | :--- | :--- | :--- | :--- |
| `rev16` | 16 episodes | Podcasts (diverse topics) | ✅ `.txt` files (~1 MB total) | ❌ Must generate |
| `ietf` | 10 sessions | IETF conference recordings | ❌ Audio only (`.opus`) | ❌ Must generate |

### Step 1: Generate Training Data (Synthetic Summaries)

Since no ground-truth summaries exist, generate them using a **large teacher
model** (much bigger than the fine-tuned student). This is standard practice
for distillation.

```bash
# On a cloud instance with an A100 (or using an API):

# Option A: Use a large local model as teacher
ollama pull qwen3.5:32b  # or qwen3.5:72b if VRAM allows
# Run each rev16 transcript through the teacher with the exact same prompt
# template, but with a much larger context window and no time pressure.

# Option B: Use an API (OpenAI, Anthropic, Google) as teacher
# Faster and higher quality, but costs money.
# Feed each 300-second window of transcript → get summary + keywords.
```

**Training data pipeline** (new script: `tools/generate_training_data.py`):
1. Load each `rev16/*.txt` transcript.
2. Chunk into 300-second windows (matching the pipeline's `window_aggregator` behavior).
3. For each window, call the teacher model with the exact production prompt template.
4. Validate output: must be valid JSON with `summary` (string) and `keywords` (3-element array).
5. Save as JSONL: `{"instruction": "<prompt>", "input": "<transcript_window>", "output": "<json_response>"}`

For `ietf` audio files, first transcribe them using Whisper, then apply the same process.

**Expected yield**: ~1 MB of transcripts ÷ ~4 KB per 300s window ≈ **250 training examples** from `rev16` alone. With `ietf` (10 sessions × ~90 min each), expect another **~180 examples**. Total: **~430 examples** — sufficient for LoRA fine-tuning of a small model.

### Step 2: Fine-Tune with LoRA/QLoRA

Fine-tuning the full model weights is unnecessary and wasteful. Use **LoRA**
(Low-Rank Adaptation) or **QLoRA** (quantized LoRA) to train only a small
adapter on top of the frozen base model.

**Recommended tools**:
- **Unsloth** — fastest LoRA training, 2× speed of HuggingFace PEFT, excellent Qwen support
- **HuggingFace TRL + PEFT** — more flexible, well-documented
- **axolotl** — config-driven, good for experimentation

**Training configuration**:
```python
# Example using Unsloth (recommended)
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-4B",
    max_seq_length=2048,
    load_in_4bit=True,        # QLoRA
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,                      # LoRA rank (16 is a good starting point)
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
)

# Train with SFTTrainer (from TRL)
# ~430 examples, 3 epochs, batch_size=4 → ~320 steps
# Estimated training time: ~15–30 min on A100, ~45–60 min on RTX 4090
```

**VRAM requirements for training**:
| Model | QLoRA Training VRAM | Inference VRAM |
| :--- | :--- | :--- |
| Qwen3.5-4B | ~12 GB | ~3 GB |
| Qwen3.5-9B | ~20 GB | ~6 GB |
| Qwen3.5-14B | ~36 GB (needs A100) | ~9 GB |

**Recommendation**: Fine-tune **Qwen3.5-4B** and **Qwen3.5-9B** — both trainable on a 24 GB GPU with QLoRA. If you want to fine-tune 14B, rent an A100 temporarily.

### Step 3: Export and Deploy the Fine-Tuned Model

**For Ollama deployment**:
```bash
# 1. Merge LoRA adapter back into the base model
python -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained('output/checkpoint-final')
model.save_pretrained_merged('merged_model', tokenizer, save_method='merged_16bit')
"

# 2. Convert to GGUF format (what Ollama uses)
pip install llama-cpp-python
python -m llama_cpp.convert merged_model --outtype q4_k_m --outfile model.gguf

# 3. Create an Ollama Modelfile
cat > Modelfile <<EOF
FROM ./model.gguf
PARAMETER num_ctx 2048
PARAMETER num_predict 150
TEMPLATE "{{ .Prompt }}"
EOF

# 4. Import into Ollama
ollama create streamind-summarizer -f Modelfile
```

**For vLLM deployment**: Simply point vLLM at the merged model directory — no conversion needed.

### Step 4: Evaluate Fine-Tuned vs. Base Models

Use `tools/eval_quality.py` to compare:

| Model | B_i (quality) | K_i (keywords) | proc_time | L_i | C_i |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Base Qwen3.5-4B | baseline | baseline | baseline | baseline | baseline |
| Fine-tuned Qwen3.5-4B | ? | ? | ? | ? | ? |
| Base Qwen3.5-9B | ? | ? | ? | ? | ? |
| Fine-tuned Qwen3.5-9B | ? | ? | ? | ? | ? |
| Base Qwen3.5-14B | ? | ? | ? | ? | ? |

**Key hypothesis**: A fine-tuned 4B model should match or exceed a generic 9B
model on `B_i` while maintaining the 4B model's speed advantage.

---

## 5. Dual-Track Inference Backend Strategy

Both tracks share the same pipeline architecture (Juturna nodes), prompt templates, and keyword handling (`_summarizer_common/keywords.py`). The only difference is the inference backend.

### Track 1: Ollama (current baseline)

**What exists**: `plugins/nodes/proc/_summarizer_llm/` — fully implemented, uses the `ollama` Python client to call a local Ollama server over HTTP.

**Setup on cloud instance**:
```bash
# Install Ollama (auto-detects CUDA)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the submission-default model
ollama pull qwen3.5:4b

# Run the pipeline
./tools/run_pipeline.sh --window 30 -s ollama-qwen3.5-4b
```

**Latency characteristics**: Ollama uses `llama.cpp` (C++ with CUDA kernels). The HTTP overhead to `127.0.0.1` is ~1–3 ms per request, negligible compared to the 1–3 second generation time. The real latency concern is model loading and KV-cache management between requests, which Ollama handles automatically.

**Pros**: Already implemented. Process isolation (separate VRAM management from Whisper). Battle-tested in the existing Dockerfile. Easy to swap models (including fine-tuned GGUF).
**Cons**: Slight overhead from HTTP serialization. No fine-grained control over inference parameters like KV-cache retention between windows.

### Track 2: Native In-Process via vLLM

**What would be built**: A new Juturna node at `plugins/nodes/proc/_summarizer_vllm/`, modeled after `_summarizer_mlx` (which already loads models in-process).

**Why vLLM**: PagedAttention provides near-optimal GPU memory utilization. Continuous batching is not needed here (single-stream pipeline), but the raw token generation speed on CUDA often exceeds `llama.cpp`/Ollama.

**Implementation sketch**:
```python
from vllm import LLM, SamplingParams

class SummarizerVllm(Node[ObjectPayload, ObjectPayload]):
    def warmup(self):
        self._llm = LLM(
            model="Qwen/Qwen3.5-4B",       # or path to fine-tuned model
            gpu_memory_utilization=0.65,     # Leave ~8 GB for Whisper ASR
            max_model_len=2048,
            enforce_eager=True,              # Avoid CUDA graph overhead for single requests
        )
        self._params = SamplingParams(max_tokens=150, temperature=0.0)
```

**Pros**: Eliminates HTTP overhead entirely. Potentially faster token generation. Full control over sampling and caching. Loads HuggingFace models directly (including LoRA adapters natively).
**Cons**: New code to write and test. vLLM eagerly pre-allocates VRAM — must carefully set `gpu_memory_utilization` to coexist with Whisper. Adds `vllm` as a heavy dependency (~2 GB install).

### Decision Protocol

Run the same test (30-second window, `youtube_15min.wav`) on the cloud GPU with both tracks. Compare:

| Metric | How to Measure |
| :--- | :--- |
| `proc_time` | Read from `results/*/window_*.json` |
| Summary quality | Run `tools/eval_quality.py` on the output |
| Peak VRAM | Watch `nvidia-smi` during the run |
| Stability | Run 3× back-to-back, check for OOM or crashes |

Choose whichever track achieves the lowest `proc_time` with `B_i ≥ 10`. If they are close, prefer Ollama for reliability (it is already tested and Dockerized).

---

## 6. Dockerfile & Submission Verification

### What Already Exists

The project already has a working Dockerfile structure:

- **`Dockerfile`**: Based on `nvidia/cuda:12.3.0-runtime-ubuntu22.04`. Installs Python 3.12, `uv`, Ollama. Pulls `qwen3.5:4b` at build time. Copies pipeline code.
- **`docker-compose.yml`**: Two services — `janus` (WebRTC gateway) and `pipeline` (the submission container). GPU passthrough configured via `deploy.resources.reservations.devices`.
- **`docker/entrypoint-pipeline.sh`**: Starts `ollama serve`, waits for readiness, then launches `./tools/run_pipeline.sh`.

### What Has Never Been Tested (Blockers #2, #3)

These are the specific risks from `submission-readiness-remaining.md` that cloud GPU testing will close:

| Blocker | Risk | How Cloud Testing Closes It |
| :--- | :--- | :--- |
| #2 CUDA build unverified | `deadsnakes` PPA, `uv sync`, Ollama install on minimal Ubuntu, build-time `ollama pull` (requires daemon mid-build) | Build the image on the cloud instance: `docker compose build` |
| #3 End-to-end container smoke | Janus → pipeline UDP/8888 → results never exercised | `docker compose up` + `send_audio.py` on cloud instance |
| #4 CUDA proc_time unknown | MLX baseline (3.10s) doesn't predict CUDA performance | Read `proc_time` from result JSON files after cloud run |

### Dockerfile Updates for Fine-Tuned Models

If the submission uses a fine-tuned model, the Dockerfile must include the model
weights. Two approaches:

**Option A (Ollama + GGUF)**: Replace the `ollama pull qwen3.5:4b` build step
with `COPY` of the GGUF file and the Modelfile, then `ollama create` during build.

**Option B (vLLM + HuggingFace)**: `COPY` the merged model directory into the
image, or use `huggingface-cli download` during the build phase.

### Verification Checklist

```bash
# On the cloud GPU instance:

# 1. Build both containers from scratch
docker compose build --no-cache

# 2. Start the full stack
docker compose up -d

# 3. Wait for pipeline to be ready (watch logs)
docker compose logs -f pipeline

# 4. Inject test audio (from the host, not inside the container)
uv run python tools/send_audio.py tests/fixtures/youtube_15min.wav

# 5. Verify results were produced
ls -la results/

# 6. Check proc_time in the output files
cat results/qwen3.5-4b/30/window_0.json | python -m json.tool

# 7. Confirm GPU utilization during the run
nvidia-smi
```

### If Switching to vLLM for Submission

The `Dockerfile` and `entrypoint-pipeline.sh` must be updated:
- Remove Ollama install and `ollama serve` startup from entrypoint.
- Add `vllm` to `pyproject.toml` dependencies.
- Add a model download step in the Dockerfile (e.g., `huggingface-cli download Qwen/Qwen3.5-4B`).
- Update `SUMMARIZER_PROFILE` env var default to point to the new vLLM profile.

---

## 7. Execution Order

```text
Phase A: Infrastructure (Day 1)
  Step 1: Rent 24 GB cloud GPU instance (RunPod, RTX A5000/4090)
  Step 2: Clone repo, install deps, verify nvidia-smi
  Step 3: Build & run Docker stack → closes blockers #2, #3
  Step 4: Record CUDA proc_time baseline with Qwen3.5-4B → closes blocker #4

Phase B: Model Exploration (Day 1–2)
  Step 5: Benchmark model ladder (4B → 9B → 14B) on same test audio
  Step 6: Scaffold _summarizer_vllm node (parallel with step 5)
  Step 7: Benchmark Ollama vs. vLLM for best model size

Phase C: Fine-Tuning (Day 2–3)
  Step 8: Transcribe ietf audio files with Whisper
  Step 9: Generate synthetic training data (teacher model on rev16 + ietf)
  Step 10: QLoRA fine-tune Qwen3.5-4B and Qwen3.5-9B
  Step 11: Export to GGUF (Ollama) and/or HuggingFace (vLLM)
  Step 12: Benchmark fine-tuned vs. base models

Phase D: Final Submission (Day 3–4)
  Step 13: Pick best model × backend combination
  Step 14: Update Dockerfile with final model weights
  Step 15: Clean-room verification: destroy instance, rent new one,
           git clone + docker compose up from scratch
```

---

## 8. Cost Estimate

| Activity | Duration | GPU | Cost |
| :--- | :--- | :--- | :--- |
| Infrastructure + Docker smoke test | ~2 hours | RTX A5000 (RunPod) | ~$0.80 |
| Model ladder benchmarking | ~3 hours | RTX A5000 (RunPod) | ~$1.20 |
| vLLM node development | ~3 hours | RTX A5000 (RunPod) | ~$1.20 |
| Training data generation (teacher model) | ~2 hours | RTX A5000 or A100 | ~$1.50 |
| QLoRA fine-tuning (2 models × ~1 hr) | ~2 hours | RTX 4090 or A100 | ~$2.00 |
| Fine-tuned model benchmarking | ~2 hours | RTX A5000 (RunPod) | ~$0.80 |
| Clean-room final verification | ~1 hour | RTX A5000 (RunPod) | ~$0.40 |
| **Total** | **~15 hours** | | **~$8.00** |

> **Tip**: Stop the instance when not actively testing. RunPod charges per-minute,
> so pausing between coding sessions saves money. Use VS Code Remote to reconnect
> instantly when you resume.

---

## 9. Expected Score Impact

| Configuration | B_i | K_i | L_i | C_i (per chunk) |
| :--- | :--- | :--- | :--- | :--- |
| Current (MLX 4B, 3.10s) | ~15 | ~5 | ~2.1 | ~22.1 |
| Base 9B on CUDA (~2.5s) | ~18 | ~5 | ~2.9 | ~25.9 |
| Base 14B on CUDA (~4s) | ~20 | ~5 | ~1.4 | ~26.4 |
| **Fine-tuned 4B on CUDA (~1.5s)** | ~20 | ~6 | ~4.7 | **~30.7** |
| **Fine-tuned 9B on CUDA (~2.5s)** | ~22 | ~6 | ~2.9 | **~30.9** |

The fine-tuned 4B model is projected to be the best overall because it combines
near-9B quality with 4B speed, maximizing both `B_i` and `L_i` simultaneously.
The + Janus bonus of 4 points brings the per-audio score to **~35**.
