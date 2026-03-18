# STREAMIND Project — Claude Agent Instructions

## Challenge Summary

STREAMIND is a Grand Challenge to build a **real-time AI meeting intelligence pipeline**. The system receives a live audio stream from a meeting, transcribes it incrementally, aggregates context, and produces structured LLM outputs (summary + keywords) with minimal latency.

Scoring: `Score = LLM-as-judge(summary quality, 0–30) − latency_penalty`

Latency directly subtracts from score. Every millisecond matters.

## Pipeline Architecture

The pipeline has 6 sequential stages, all implemented as **Juturna nodes**:

1. **Audio Reception** — Receive Opus-encoded mono RTP stream via WebRTC (through Janus gateway)
2. **Incremental ASR** — Transcribe short, consecutive, partially overlapping audio chunks
3. **Novel Chunk Extraction** — Deduplicate overlapping content between consecutive chunks; extract only the new portion
4. **Window Aggregation** — Accumulate transcript into 300-second rolling context windows
5. **Summarization** — LLM produces one summary + exactly 3 keywords per window
6. **Transmission** — Store result locally and POST to the challenge destination endpoint

## Key Frameworks

### Juturna

- Open-source Python framework for real-time AI data pipeline prototyping
- Node-based composable architecture
- **All pipeline stages must be implemented as Juturna nodes**
- Repo: <https://github.com/meetecho/juturna>

### Janus

- Open-source WebRTC server used for audio delivery to the pipeline
- Repo: <https://github.com/meetecho/janus-gateway>

## Scoring Constraints

- Summary quality scored 0–30 by an LLM-as-judge
- Latency is subtracted directly from the score — minimize end-to-end processing time
- Each window output **must include exactly 3 keywords**
- Missing keywords or malformed output will hurt the score

## Key References

- Full challenge specification: `docs/CHALLENGE.md`
- Juturna framework: <https://github.com/meetecho/juturna>
- Janus WebRTC server: <https://github.com/meetecho/janus-gateway>
