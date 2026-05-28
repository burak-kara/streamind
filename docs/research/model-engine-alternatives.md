# Model & Inference Engine Research — May 2026

> Goal: best summarization quality (B_i) under lowest possible latency (L_i).
> Constraint: single RTX Pro 4500 (24 GB VRAM), real-time pipeline, submission Docker with no network.

## Current Baseline

- **Model:** Qwen3.5-4B (multimodal, `Qwen3_5ForConditionalGeneration`)
- **Engine:** vLLM in-process (`LLM()` class)
- **Score:** avg C = 34.29 (Final 38.29 with Janus +4 / 45)
- **Known issues:** multimodal arch wastes VRAM on vision encoder; proc_time 1-3s

---

## 1. Inference Engines

Two independent axes: what model you run, and how fast you run it.

| Engine | In-process API | Latency vs vLLM | Setup cost | Notes |
|--------|---------------|-----------------|-----------|-------|
| **[SGLang](https://docs.sglang.io/docs/basic_usage/offline_engine_api)** | `sgl.Engine(model_path=...)` | ~29% higher throughput, lower TTFT | `pip install sglang` | RadixAttention; drop-in replacement for vLLM |
| **[TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/)** | Python API | 15-30% over vLLM, 70% over llama.cpp | 28-min compilation per model | Best raw speed; heavy setup |
| **vLLM** (current) | `LLM()` | baseline | already integrated | Widest model support, no compilation |
| **HF Transformers** | `model.generate()` | N/A (for enc-dec only) | native | Only needed for BART/T5/Pegasus path |

### SGLang offline API (drop-in for vLLM)

```python
import sglang as sgl

llm = sgl.Engine(model_path="./models/qwen3-30b-a3b-awq")
outputs = llm.generate(prompts, {"temperature": 0.3, "max_new_tokens": 150})
```

**Recommendation:** SGLang swap = ~29% latency win for minimal code change. TensorRT-LLM = more speed but compilation overhead + narrower model support.

---

## 2. Model Architectures

### Option A: Decoder-only instruction LLM (current approach)

| Model | Params (active) | VRAM (AWQ int4) | Est. proc_time | JSON + keywords | License |
|-------|----------------|----------------|----------------|-----------------|---------|
| Qwen3.5-4B *(current)* | 4B dense | ~10 GB fp16 | 1-3s | native | Apache 2.0 |
| [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen) AWQ | 3B active / 30B total | ~16-17 GB | 0.8-2s | native | Apache 2.0 |
| [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) AWQ | 3B active / 35B total | ~17-22 GB | 0.8-2s | native | Apache 2.0 |
| Phi-3.5-mini-instruct | 3.8B dense | ~8 GB fp16 | 0.5-1.5s | native | MIT |
| Mistral-7B-Instruct-v0.3 | 7B dense | ~14 GB fp16 | 1-3s | native | Apache 2.0 |

**Pros:** reliable JSON `{summary, keywords[3]}`, strong factual consistency, proven path.
**Cons:** proc_time 1-3s → L_i ≈ 2-6 (out of 10 max).

#### GGUF warning

`unsloth/Qwen3.6-35B-A3B-MTP-GGUF` exists but **GGUF + vLLM MoE is experimental and flaky**. vLLM/SGLang native path uses AWQ Marlin kernels — mature, fast. Always prefer AWQ/GPTQ over GGUF for vLLM/SGLang.

Also: UD-Q4_K_M is 22.7 GB on disk → only ~1.3 GB left for KV cache on 24 GB GPU. OOM risk. We already got burned by `cyankiwi/Qwen3.5-27B-AWQ` (claimed int4, actually 26 GB on-disk).

**Always verify with `du -sh ./models/<name>` before trusting HF-reported sizes.**

#### MoE advantage for this task

MoE models like Qwen3-30B-A3B activate only ~3B params per token but draw from 30B total capacity. This means:
- Decode speed close to a 3B dense model
- Quality closer to a 30B dense model
- Sweet spot for our "quality under latency" goal

### Option B: Encoder-decoder summarization models

| Model | Params | VRAM (fp16) | Est. proc_time | JSON/keywords | Max input | License |
|-------|--------|------------|----------------|---------------|-----------|---------|
| [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) | 400M | ~1.5 GB | <100ms | **NO** | 1024 tok | MIT |
| [philschmid/bart-large-cnn-samsum](https://huggingface.co/philschmid/bart-large-cnn-samsum) | 400M | ~1.5 GB | <100ms | **NO** | 1024 tok | MIT |
| [google/pegasus-large](https://huggingface.co/google/pegasus-large) | 568M | ~2 GB | <100ms | **NO** | 1024 tok | Apache 2.0 |
| [google/flan-t5-large](https://huggingface.co/google/flan-t5-large) | 780M | ~3 GB | <150ms | partial | 512 tok* | Apache 2.0 |
| [allenai/led-large-16384](https://huggingface.co/allenai/led-large-16384) | 460M | ~1.8 GB | <100ms | **NO** | **16K tok** | Apache 2.0 |

*T5/FLAN-T5 default context is 512 tokens; can be extended but quality degrades.

**Pros:**
- **10-30x faster inference.** proc_time <200ms → L_i ≈ 9.5 (vs 3-6 for decoder-only LLM).
- No vLLM/SGLang needed — plain HuggingFace `model.generate()`.
- Tiny VRAM footprint — leaves room for larger ASR model or other components.

**Cons:**
- No JSON output format — need Python wrapper to assemble `{summary, keywords[3]}`.
- No keyword extraction — need separate tool (RAKE, TF-IDF, KeyBERT, ~10ms).
- Trained on **CNN/DailyMail news**, not meeting transcripts. `bart-large-cnn-samsum` is dialogue-trained but still not real meetings.
- BART/Pegasus max context = 1024 tokens. Our 300s window ≈ 750-1000 words ≈ ~1000 tokens. **Tight.** LED handles 16K — safer for longer windows.
- **Risk: B_i drops below 10 → lose L_i entirely → net negative despite speed.**

#### Longformer / T5 / Pegasus — honest assessment

These are **2019-2020 era** models. "Mostly recommended" in search results reflects old blog SEO, not 2025+ reality.

- **Longformer (LED):** sparse attention for long documents. Our window is ~1000 tokens — sparse attention has no advantage below 4K tokens. Only useful if we switch to much larger windows.
- **T5/FLAN-T5:** text-to-text framework is flexible but max 512 tokens is too short for 300s windows without chunking. FLAN-T5 instruction-tuned variant is better but still limited context.
- **Pegasus:** best ROUGE on CNN/DailyMail, but domain mismatch with meeting transcripts. No instruction following.

None of these produce structured JSON or extract keywords natively. All need post-processing.

Recent research (Oct 2025) shows encoder-decoder models **can** be competitive with decoder-only LLMs after task-specific tuning, and dominate the inference efficiency Pareto frontier. But we have no time to fine-tune BART/Pegasus on meeting data before submission.

### Option C: Hybrid (recommended to A/B test)

Use both architectures, compare empirically:

**Fast path (encoder-decoder):**
1. `bart-large-cnn-samsum` or `led-large-16384` for summary (~100ms)
2. KeyBERT or RAKE for keyword extraction (~10ms)
3. Python wrapper assembles JSON

**Total: ~110ms → L_i = 9.47**

**Quality path (decoder-only LLM):**
1. Qwen3-30B-A3B AWQ on SGLang for summary + keywords (~1-2s)

**Total: ~1.5s → L_i = 4.72**

---

## 3. The Scoring Math

```
C_i = B_i + K_i + L_i
L_i = 10 * exp(-0.5 * proc_time)    [ONLY when B_i >= 10; otherwise L_i = 0]
Final = mean(C_i) + Janus_bonus(+4)

B_i range: [5, 25]     (5 dimensions × [1,5] each)
K_i range: [-6, +6]    (+2 per relevant keyword, -2 per irrelevant, 3 required)
L_i range: [0, 10]     (exponential decay; B_i >= 10 gate)
Max C_i: 41 (+ 4 Janus = 45)
```

### L_i sensitivity table

| proc_time | L_i   | vs 2s baseline |
|-----------|-------|----------------|
| 0.1s      | 9.51  | +5.83          |
| 0.2s      | 9.05  | +5.37          |
| 0.5s      | 7.79  | +4.11          |
| 1.0s      | 6.07  | +2.39          |
| 2.0s      | 3.68  | baseline       |
| 3.0s      | 2.23  | -1.45          |
| 5.0s      | 0.82  | -2.86          |

### Break-even analysis

Switching from 2s LLM to 0.1s BART gains +5.83 L_i per window.
This is worth it **only if B_i doesn't drop more than 5.83 points.**

| Scenario | B_i (LLM) | B_i (BART) | ΔB | ΔL | ΔC (net) | Verdict |
|----------|-----------|------------|-----|-----|----------|---------|
| BART matches | 20 | 20 | 0 | +5.8 | **+5.8** | Switch |
| BART slightly worse | 20 | 16 | -4 | +5.8 | **+1.8** | Switch |
| BART mediocre | 20 | 12 | -8 | +5.8 | **-2.2** | Stay LLM |
| BART below cliff | 20 | 9 | -11 | -3.7* | **-14.7** | Disaster |

*Below cliff: L_i_BART = 0 vs L_i_LLM = 3.68

**B_i = 10 is the cliff.** Below it, L_i = 0 regardless of speed.

---

## 4. Action Plan

### Phase 1 — Low-hanging fruit (do now)

- [ ] **SGLang swap:** replace `vllm.LLM` → `sgl.Engine` in `summarizer_vllm.py`. ~29% latency win, minimal code change. Test on uni-lab.
- [ ] **Model swap:** Qwen3.5-4B → Qwen3-30B-A3B-Instruct-2507 AWQ (not GGUF). Fetch with `tools/fetch_models.sh`, verify `du -sh` < 20 GB, update profile JSON.

### Phase 2 — A/B test encoder-decoder (evaluate before committing)

- [ ] **Build BART summarizer node:** new `_summarizer_bart/` or make existing node configurable. Load `bart-large-cnn-samsum` via HF transformers. Add KeyBERT for keywords. Wrapper produces `{summary, keywords[3]}`.
- [ ] **Run both pipelines on same audio on uni-lab.** Judge both with Mistral-Small-24B. Compare B_i, K_i, L_i, C_i.
- [ ] **Decision gate:** if BART B_i ≥ 15, switch. If B_i 10-15, marginal — test more. If B_i < 10, stay decoder-only.

### Phase 3 — Optimize winner

- [ ] If decoder-only wins: TensorRT-LLM compilation for extra 15-30% speed
- [ ] If encoder-decoder wins: fine-tune on meeting transcripts (rev16 or MeetingBank) for B_i lift
- [ ] Either path: prompt/sampling tune (M2 milestone)

---

## 5. Models NOT to use

| Model | Reason |
|-------|--------|
| Any GGUF format | vLLM/SGLang MoE GGUF support flaky; use AWQ/GPTQ |
| Qwen3.5-4B (current) | Multimodal arch wastes VRAM on vision encoder |
| Any model > 20 GB on-disk | OOM risk on 24 GB after KV cache + activations |
| Ollama-served anything | Project constraint: no daemons |
| Raw T5-base/large | 512 token context too short for 300s windows |

---

## References

- [SGLang Offline Engine API](https://docs.sglang.io/docs/basic_usage/offline_engine_api)
- [SGLang vs vLLM 2026 Comparison](https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison)
- [vLLM vs TensorRT-LLM vs SGLang H100 Benchmarks](https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/)
- [Best Open-Source LLMs for Summarization 2026](https://www.siliconflow.com/articles/en/best-open-source-llms-for-summarization)
- [Qwen Speed Benchmarks](https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html)
- [Best Inference Engines 2026](https://www.yottalabs.ai/post/best-llm-inference-engines-in-2026-vllm-tensorrt-llm-tgi-and-sglang-compared)
- [Qwen3.6-35B-A3B HuggingFace](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [TensorRT-LLM RTX 4090 Benchmarks](https://medium.com/write-a-catalyst/i-tested-every-local-llm-framework-so-you-dont-have-to-fbdb31d1aca7)
- [BART vs T5 vs Pegasus Comparative Study](https://www.mdpi.com/1999-5903/17/9/389)
