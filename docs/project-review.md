# STREAMIND Project Review

**Date:** 2026-04-10  
**Scope:** Full assessment against challenge requirements, Juturna framework alignment, latency analysis, dead code, and correctness issues.

---

## Executive Summary

The pipeline is functionally complete — all 6 required stages exist and are correctly wired in `pipelines/config.json`. However, the project has **one showstopper latency problem, one dead-code node that cannot be used as-is, and several correctness gaps** that will cost points on the scoring formula `C_i = S_i − latency_i`.

Measured latency from test runs: **30–63 seconds per window**. Since the maximum score `S_i` is 30, the current pipeline will produce zero or negative scores on every chunk. Every fix below is ordered by its impact on this formula.

---

## 1. Latency Analysis

### 1.1 Measured vs. Required

| Window | Latency (measured) | Max possible score |
|--------|--------------------|--------------------|
| `window_0.json` | 30.09 s | 0.0 (30 − 30) |
| `window_1.json` | 63.04 s | −33 (negative) |
| `window_2.json` | ~30–60 s (inferred) | 0 or negative |

Target: **sub-30 seconds**, ideally sub-10 seconds.

### 1.2 What is Being Measured

`trigger_time` is set in `novel_extractor.py:53` (`time.time()`) and propagated via the payload. `window_aggregator._flush()` passes it through to the summarizer. `summarizer_llm.py:107` computes `latency = time.time() - trigger_time`.

This means **latency = wall time from when the last ASR chunk crossed the window boundary → LLM result delivered**. This is a reasonable definition consistent with the challenge's intent (time to produce the summary after the window closes), but it means the LLM inference time is the sole bottleneck being measured.

### 1.3 Root Cause: LLM Inference Time

`qwen3.5:9b-16k` on Apple Silicon (no NVIDIA GPU) takes 30–63 seconds per inference. The model uses chain-of-thought `<think>` tokens internally, which significantly inflate response time.

**Fix options, ordered by impact:**

| Option | Expected speedup | Risk |
|--------|-----------------|------|
| Switch to `qwen3:1.7b` | ~4–5× faster | Slight quality drop |
| Switch to `qwen3:0.6b` | ~10–12× faster | Larger quality drop |
| Add `/no_think` header to prompt | ~2× faster (same model) | None |
| Add stop sequence `</think>` | cuts thinking early | May truncate output |
| Use `num_ctx` limit in Ollama options | reduces KV-cache overhead | May truncate context |

The prompt currently says nothing to suppress thinking. Add `/no_think` as the first line of `summarize_prompt.txt`, which is the official qwen3 instruction to skip the reasoning pass:

```
/no_think
You are a meeting summarization assistant...
```

**File:** `plugins/nodes/proc/_summarizer_llm/summarize_prompt.txt:1`

### 1.4 VAD is Disabled — Transcribing Silence

`transcriber_whisper.py:67` sets `vad_filter=False`. On silent segments, faster-whisper will hallucinate filler words (music, laughter, etc.), adding noise to the transcript. This increases token count → longer LLM prompt → slower inference.

**Fix:** Set `vad_filter=True`. This adds ~1ms per chunk but can reduce transcript size by 20–40% on meetings with pauses.

**File:** `plugins/nodes/proc/_transcriber_whisper/transcriber_whisper.py:67`

### 1.5 No Timeout on LLM Call

`summarizer_llm.py:93` calls `self._client.chat(...)` with no timeout. A stalled Ollama instance will block the pipeline indefinitely. Add `options={"num_predict": 512}` to cap output length and avoid runaway generation.

**File:** `plugins/nodes/proc/_summarizer_llm/summarizer_llm.py:93`

---

## 2. Dead Code: `transcriber_whispy`

The `_transcriber_whispy` node was provided by Meetecho (author: Antonio Bevilacqua) and is **not wired into `config.json`**. It cannot be used as a drop-in replacement for `_transcriber_whisper` because:

### 2.1 Incompatible Output Format

`transcriber_whispy.update()` outputs `payload['transcript']` as a **list of word dicts** with keys `word`, `start`, `end`, `probability`. `novel_extractor.py:43` calls `message.payload.get("transcript", "")` and calls `.split()` on the result. This will crash with a list input.

### 2.2 Model Loaded in `__init__` (Framework Violation)

**File:** `plugins/nodes/proc/_transcriber_whispy/transcriber_whispy.py:70–72`

```python
self._model = WhisperModel(
    model_name, local_files_only=self._only_local, device=device
)  # ← runs at construction time, blocks pipeline startup
```

Juturna's node development guide (section "Node Class Template") requires heavy resources to load in `warmup()`, not `__init__`. Loading a Whisper model at construction time blocks the pipeline initialization thread.

### 2.3 Missing Lifecycle Methods

`transcriber_whispy` has no `start()` or `stop()`. The Juturna template requires both to call `super()`. Without `super().start()`, the node's worker thread is never launched; the node silently drops all messages.

**Missing:**
- `start(self): super().start()`
- `stop(self): super().stop()`
- `configure(self): pass`

### 2.4 `time.sleep()` Calls in `destroy()`

**File:** `plugins/nodes/proc/_transcriber_whispy/transcriber_whispy.py:242, 248, 254`

Three blocking `time.sleep(1)` calls during model teardown add 3+ seconds of shutdown latency. This affects the last window, which must be drained through the summarizer during `stop()`.

### 2.5 `device = "cuda"` on Apple Silicon

**File:** `plugins/nodes/proc/_transcriber_whispy/config.toml:9`

Hardcoded `device = "cuda"` will fail silently or crash on Apple Silicon. Should be `"auto"`.

### 2.6 Depends on Non-Standard Message Metadata

`transcriber_whispy.py:110` reads `message.meta['silence']` and line 177 reads `message.meta['speech_timestamps']`. These metadata keys are set by the built-in `audio_rtp` node but not by `audio_file`. The node is tightly coupled to `audio_rtp`.

### 2.7 Commented-Out Code

`transcriber_whispy.py:81`: `# self._data = collections.deque(maxlen=buffer_size)` — orphaned comment from a refactor.

**Verdict:** This node is dead code in its current state. It requires significant rework before it can be connected into the pipeline. If Meetecho's word-level output format is needed for a future feature, it should be kept but clearly marked as non-functional. Otherwise, remove it to avoid confusion.

---

## 3. Framework Alignment Issues

### 3.1 `window_aggregator` Hard-Codes `version=1`

**File:** `plugins/nodes/proc/_window_aggregator/window_aggregator.py:42`

```python
out = Message[ObjectPayload](creator=self.name, version=1, payload=payload)
```

Every window message gets `version=1`, losing message lineage. The Juturna template shows `version=message.version` to propagate identity through the pipeline. The window flush is triggered by a message (line 57), so `message.version` is available:

```python
out = Message[ObjectPayload](creator=self.name, version=message.version, payload=payload)
```

For the `stop()`-triggered flush (line 26), there is no incoming message — the version reset is unavoidable there, but line 57's flush should propagate it.

### 3.2 `window_aggregator._flush()` Called from `stop()` With No Message Context

**File:** `plugins/nodes/proc/_window_aggregator/window_aggregator.py:26`

When the pipeline stops, `_flush(time.time())` is called with no access to the last received message. This means `timers_from` cannot be set on the final window's output message, and the timer chain is broken for the last window. This is a structural limitation — `_flush` needs to optionally accept the triggering message:

```python
def _flush(self, wall_time: float, source_message=None):
    ...
    out = Message[ObjectPayload](
        creator=self.name,
        version=source_message.version if source_message else 1,
        payload=payload,
        timers_from=source_message,
    )
```

### 3.3 `_queue` Access in `stop()` Is Private API

**Files:**
- `plugins/nodes/proc/_summarizer_llm/summarizer_llm.py:72`
- `plugins/nodes/sink/_result_transmitter/result_transmitter.py:31`

Both nodes drain `self._queue` directly in `stop()`. `_queue` is a private attribute of the Juturna `Node` base class. This is fragile against Juturna version updates. If Juturna provides a graceful drain mechanism, use it. If not, document this as a known coupling.

---

## 4. Configuration Issues

### 4.1 `destination_endpoint` Is Empty

**File:** `pipelines/config.json:77`

```json
"destination_endpoint": ""
```

Results are only stored locally. Must be set to the challenge POST URL before submission. The code handles this gracefully (skips POST when empty), so local testing works correctly — but this **must not be forgotten before submission**.

### 4.2 `encoding_clock_chan` Stereo vs. `channels: 1` Mono

**File:** `pipelines/config.json:21–22`

```json
"channels": 1,
"encoding_clock_chan": "opus/48000/2"
```

`opus/48000/2` declares a stereo Opus stream (2 channels), but `channels: 1` tells the node to output mono. The built-in `audio_rtp` node should downmix, but this mismatch is worth verifying against the actual Janus stream parameters. If Janus sends a mono stream, change to `opus/48000/1`.

### 4.3 Window Duration Is 300s But ASR Chunks Are 5s Each

With `chunk_duration: 5.0` and `overlap_duration: 1.0`, the window accumulates **60+ ASR chunks** before flushing. Each chunk carries its full word-level context. The transcript for a 300s window can easily reach 3,000–5,000 words. The LLM must process all of it — this is the secondary driver of latency after model size.

Consider whether the aggregated transcript should be summarized incrementally (rolling summary passed to LLM as context) rather than always sending the raw full transcript. This is an architectural improvement, not a bug.

---

## 5. Test Coverage Gaps

| Test file | Status |
|-----------|--------|
| `test_audio_chunker.py` | Present — basic coverage |
| `test_novel_extractor.py` | Present — basic coverage |
| `test_window_aggregator.py` | Present — good coverage |
| `test_summarizer_llm.py` | Present — mocked Ollama, covers keywords enforcement |
| `test_result_transmitter.py` | Present |
| `test_pipeline_integration.py` | **Empty (`pass`)** — zero integration testing |

Critical gaps:
- No test for `transcriber_whisper` (requires model download, but can be mocked)
- `test_pipeline_integration.py` is a stub — no end-to-end validation exists
- No test verifies that `trigger_time` → `latency` computation is correct
- No test for the `stop()`-triggered window flush path

---

## 6. Minor Code Quality Issues

### 6.1 Empty `configure()` and `set_on_config()` Across All Nodes

Every node has `def configure(self): pass` and `def set_on_config(self, prop, value): pass` as stubs. Per the Juturna template these are valid stubs, but `set_on_config` is only needed if runtime reconfiguration is supported. If it's never used, it can be omitted entirely (the base class no-ops it).

### 6.2 `novel_extractor.py` Sets `wall_time = time.time()`

**File:** `plugins/nodes/proc/_novel_extractor/novel_extractor.py:53`

This timestamp is consumed by `window_aggregator` as `trigger_time` to measure latency. It is set when the *novel extractor* processes, not when the *audio chunk ends*. For a real-time stream these are nearly identical (~milliseconds apart), so this is acceptable, but it should be documented as the intended measurement anchor.

### 6.3 `transcriber_whisper` Does Not Use `timeit` on the Hot Path

**File:** `plugins/nodes/proc/_transcriber_whisper/transcriber_whisper.py:63–69`

The model transcription is not wrapped in `with message.timeit(...)`. This means per-node timing data is absent from the message timer chain, making it impossible to profile where latency comes from using Juturna's built-in observability. Low priority but useful for future optimization.

---

## 7. Summary of Issues by Priority

| # | Severity | Issue | File | Line(s) |
|---|----------|-------|------|---------|
| 1 | **CRITICAL** | LLM inference 30–63s exceeds max score (30) | summarizer_llm.py | 93 |
| 2 | **CRITICAL** | qwen3 think-mode not suppressed; add `/no_think` | summarize_prompt.txt | 1 |
| 3 | **HIGH** | `transcriber_whispy` missing `start()`/`stop()` — node would silently drop messages | transcriber_whispy.py | 29 |
| 4 | **HIGH** | `transcriber_whispy` loads model in `__init__` — blocks pipeline startup | transcriber_whispy.py | 70–72 |
| 5 | **HIGH** | `transcriber_whispy` output format incompatible with `novel_extractor` — crash if wired in | transcriber_whispy.py | 160 |
| 6 | **MEDIUM** | `vad_filter=False` — silence hallucinations inflate transcript + LLM input | transcriber_whisper.py | 67 |
| 7 | **MEDIUM** | `window_aggregator` hard-codes `version=1` — breaks message lineage | window_aggregator.py | 42 |
| 8 | **MEDIUM** | `window_aggregator._flush()` drops timer chain on `stop()` path | window_aggregator.py | 26, 31 |
| 9 | **MEDIUM** | `_queue` accessed directly in `stop()` — private API coupling | summarizer_llm.py:72, result_transmitter.py:31 | — |
| 10 | **MEDIUM** | `transcriber_whispy` has 3× `time.sleep(1)` in destroy — last window latency | transcriber_whispy.py | 242, 248, 254 |
| 11 | **MEDIUM** | `transcriber_whispy` device=cuda — fails on Apple Silicon | config.toml | 9 |
| 12 | **LOW** | No LLM call timeout — stalled Ollama blocks pipeline forever | summarizer_llm.py | 93 |
| 13 | **LOW** | `encoding_clock_chan` stereo vs `channels` mono mismatch | config.json | 21–22 |
| 14 | **LOW** | `test_pipeline_integration.py` is empty | tests/ | — |
| 15 | **LOW** | `destination_endpoint` empty — must be set before submission | config.json | 77 |
| 16 | **INFO** | `transcriber_whisper` hot path not wrapped in `timeit` | transcriber_whisper.py | 63 |
| 17 | **INFO** | Commented-out code in `transcriber_whispy.py:81` | transcriber_whispy.py | 81 |

---

## 8. Recommended Fix Order

1. **Add `/no_think` to `summarize_prompt.txt`** — 1-line change, potentially halves latency.
2. **Benchmark smaller model** (`qwen3:1.7b`) — swap `model_name` in `config.json:68`, run pipeline, compare score.
3. **Enable `vad_filter=True`** in `transcriber_whisper.py:67` — reduces transcript size and LLM input.
4. **Add `num_predict` cap** to Ollama call options in `summarizer_llm.py:93` — prevents runaway generation.
5. **Fix `window_aggregator` version propagation** — `version=message.version` at line 42.
6. **Refactor `_flush()` to accept optional source message** — restore timer chain on stop-path.
7. **Either fix `transcriber_whispy` completely or move it to a `dead_code/` folder** — currently misleading.
8. **Set `destination_endpoint`** before challenge submission.
9. **Fill in `test_pipeline_integration.py`** — run pipeline against test fixture WAV and verify output.
