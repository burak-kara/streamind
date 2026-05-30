"""Shared LLM-as-judge scoring used by the offline eval tools.

Single source of truth for the judge prompt, the response parser, the B/K/L/C
math defined by the challenge, and the Janus bonus. Keeping these together
prevents `tools/eval_quality.py` and `tools/eval_multi_judge.py` from drifting.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, asdict


# Flat +4 added to the final source score for using Janus as the audio source
# (per CHALLENGE.md). Part of the scoring contract — kept here so both offline
# eval tools read one value.
JANUS_BONUS = 4.0


LIKERT_CRITERIA = (
    "factual_consistency",
    "relevance",
    "coherence",
    "fluency",
    "conciseness",
)

JUDGE_PROMPT = """You are a strict evaluator for meeting summaries. Given the TRANSCRIPT and the CANDIDATE JSON (summary + 3 keywords), output ONLY a raw JSON object (no markdown) with these fields:

- "factual_consistency": integer 1-5 — does the summary state only facts present in the transcript?
- "relevance": integer 1-5 — does the summary capture the main point of the transcript?
- "coherence": integer 1-5 — is the summary logically ordered and readable?
- "fluency": integer 1-5 — grammatical, natural English?
- "conciseness": integer 1-5 — short and specific, no padding?
- "keyword_relevance": array of exactly 3 booleans — is each keyword a specific, relevant term from the transcript (not a generic word like "meeting", "discussion", "topic")?

TRANSCRIPT:
{transcript}

CANDIDATE:
{candidate}
"""


@dataclass
class JudgeScore:
    b_breakdown: dict[str, int]
    b: int
    keyword_flags: list[bool]
    k: int
    l: float
    c: float

    def to_dict(self) -> dict:
        d = asdict(self)
        # Use the challenge-style upper-case names in serialised output so
        # downstream tooling (and humans) can find them quickly.
        return {
            "B_breakdown": d["b_breakdown"],
            "B": d["b"],
            "keyword_flags": d["keyword_flags"],
            "K": d["k"],
            "L": d["l"],
            "C": d["c"],
        }


def strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def build_prompt(transcript: str, summary: str, keywords: list[str]) -> str:
    candidate = json.dumps(
        {"summary": summary, "keywords": list(keywords)},
        ensure_ascii=False,
    )
    return JUDGE_PROMPT.format(transcript=transcript, candidate=candidate)


def parse_judge_response(content: str) -> tuple[dict[str, int], list[bool]]:
    """Parse a judge LLM response into (likert_scores, keyword_flags).

    Tolerates markdown fences and trailing commentary. Raises ValueError if no
    JSON object can be located in the output.
    """
    cleaned = strip_fences(content)
    obj_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if obj_match is None:
        raise ValueError(f"No JSON object found in judge output: {cleaned[:120]!r}")
    judged = json.loads(obj_match.group(0))

    b_breakdown = {}
    for crit in LIKERT_CRITERIA:
        try:
            v = int(judged.get(crit, 0))
        except (TypeError, ValueError):
            v = 0
        # Clamp to the valid Likert range — defends against models emitting 0/6.
        v = max(1, min(5, v)) if v > 0 else 0
        b_breakdown[crit] = v

    flags_raw = judged.get("keyword_relevance") or []
    keyword_flags = [bool(f) for f in flags_raw][:3]
    while len(keyword_flags) < 3:
        keyword_flags.append(False)

    return b_breakdown, keyword_flags


def compute_score(
    b_breakdown: dict[str, int],
    keyword_flags: list[bool],
    proc_time: float,
) -> JudgeScore:
    """Apply the challenge formula: C = B + K + L, with L gated on B >= 10."""
    b = sum(b_breakdown.values())
    k = sum(2 if f else -2 for f in keyword_flags)
    l_val = 10.0 * math.exp(-0.5 * proc_time) if b >= 10 else 0.0
    c = b + k + l_val
    return JudgeScore(
        b_breakdown=b_breakdown,
        b=b,
        keyword_flags=keyword_flags,
        k=k,
        l=l_val,
        c=c,
    )
