# STREAMIND Compliance & Gap Review

**Date:** 2026-04-10  
**Scope:** `docs/CHALLENGE.md`, `docs/documentation/juturna/*`, `docs/documentation/janus/*` vs current implementation.

## Executive outcome

The pipeline is structurally complete and follows the required 6-stage flow, but it is **not challenge-competitive yet** because measured latency is **30.09s, 46.60s, and 63.04s** (`results/window_0.json`, `window_2.json`, `window_1.json`).  
Given challenge scoring `C_i = S_i - latency_i` (max `S_i=30`), current runs produce near-zero or negative chunk scores.

---

## 1) Requirement alignment (Challenge)

| Requirement (`docs/CHALLENGE.md`) | Current state | Status | Evidence |
|---|---|---|---|
| Audio reception via WebRTC/RTP (Opus, mono) | `audio_rtp` source configured | **Partial** | `pipelines/config.json` (`mark: audio_rtp`, `channels:1`), but `encoding_clock_chan:"opus/48000/2"` conflicts with mono target |
| Incremental ASR on overlapping chunks | Implemented | **Met** | `chunk_duration:5.0`, `overlap_duration:1.0`, `transcriber_whisper` |
| Novel chunk extraction | Implemented | **Met** | `plugins/nodes/proc/_novel_extractor/novel_extractor.py` |
| 300s window aggregation | Implemented | **Met** | `window_duration:300.0` in config |
| Structured summary + 3 keywords | Implemented and enforced | **Met** | `_summarizer_llm.py` `_ensure_three_keywords` |
| Store locally + POST to endpoint | Local write works; POST optional | **Partial** | `destination_endpoint` is empty in config |
| Low-latency competitive output | Not met | **Missing** | Result latencies >30s |

---

## 2) Juturna alignment issues

1. `window_aggregator` breaks message lineage by hardcoding `version=1` in emitted messages.  
   - File: `plugins/nodes/proc/_window_aggregator/window_aggregator.py` (line with `Message(... version=1 ...)`)
2. Final flush path (`stop() -> _flush(time.time())`) has no `timers_from`, so observability/timer chain is incomplete for last window.
3. `summarizer_llm` and `result_transmitter` directly drain `self._queue` in `stop()` (private API coupling to Juturna internals).

Reference: `docs/documentation/juturna/node-development.md` expects lifecycle (`start/stop`) and timer propagation with `timers_from=message`.

---

## 3) Janus/readiness issues

1. RTP source exists and appears correctly wired to a Janus-fed stream port (`127.0.0.1:8888`), but there is no real integration validation.
2. Potential format mismatch:
   - `channels: 1` with `encoding_clock_chan: "opus/48000/2"` in `pipelines/config.json`.
3. Integration test is a stub:
   - `tests/test_pipeline_integration.py` contains only `pass`.

---

## 4) Latency root causes (sub-30s blocker)

### Primary bottleneck: summarizer LLM call

- Current model is `qwen3.5:9b-16k` (`pipelines/config.json`).
- Prompt currently does not disable qwen3 reasoning mode (`plugins/nodes/proc/_summarizer_llm/summarize_prompt.txt`).
- This is the dominant contributor to 30–63s per-window latency in current outputs.

### Secondary contributors

1. `vad_filter=False` in ASR (`_transcriber_whisper/transcriber_whisper.py`) increases noisy transcript volume in pauses/silence.
2. Full 300s transcript is sent as one prompt (`full_transcript` join in `_window_aggregator.py`), increasing token load.
3. No generation cap/options in main summarize call (`_summarizer_llm.py`), increasing long-response risk.

---

## 5) Dead code / misleading code surfaces

## `_transcriber_whispy` is effectively dead/unusable in current pipeline

- Not wired in `pipelines/config.json`.
- Incompatible output contract with `novel_extractor`:
  - `transcriber_whispy` emits `transcript` as list of word dicts.
  - `novel_extractor` expects string and does `.split()`.
- Model is loaded in `__init__` (heavy init path), and config defaults `device="cuda"` in its `config.toml`, which is incompatible for many non-CUDA setups.
- Contains teardown sleeps in destroy path that add avoidable delay.

Recommendation: either fully rehabilitate this node or move/mark as deprecated dead code to avoid accidental use.

---

## 6) Reliability and quality issues

1. No real end-to-end test coverage (`test_pipeline_integration.py` stub).
2. Empty `destination_endpoint` means no challenge submission transport unless manually changed.
3. Broad fallback in summarizer on LLM failure can hide model/endpoint issues by returning generic summaries.

---

## 7) Prioritized fix list

## P0 (do first)

1. **Reduce summarization latency immediately**
   - Add `/no_think` as first line in `plugins/nodes/proc/_summarizer_llm/summarize_prompt.txt`.
   - Re-benchmark latency on same sample audio.
2. **Use faster summarizer model if still >30s**
   - Change `model_name` to smaller qwen3 variant in `pipelines/config.json`.
3. **Set real `destination_endpoint` before submission**
   - `pipelines/config.json`.

## P1

1. Turn on VAD in `transcriber_whisper` (`vad_filter=True`).
2. Fix message version/timer propagation in `window_aggregator` (`version=message.version`, preserve `timers_from` where possible).
3. Add cap/options on LLM generation in `_summarizer_llm.py` to prevent long generations.

## P2

1. Implement a real integration test in `tests/test_pipeline_integration.py`.
2. Remove or deprecate `_transcriber_whispy` until it is contract-compatible.
3. Reconcile RTP channel config with actual Janus stream (`opus/48000/1` vs `/2`).

---

## 8) Suggested target state for sub-30s

Minimum tactical path:

1. `/no_think` prompt directive.
2. Smaller summarizer model if needed.
3. `vad_filter=True`.
4. Keep prompt short and strictly schema-focused.

This combination is the highest-probability route to consistent sub-30s latency while preserving required output structure.
