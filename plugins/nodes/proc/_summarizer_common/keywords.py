"""Keyword post-processing for the vLLM summarizer node.

One place to tune the rules that defend K_i: banned-keyword filtering,
transcript-derived backfill, and exact 3-keyword output. Loaded by the
summarizer via importlib so the helpers stay editable without touching
the node implementation.
"""

from __future__ import annotations

import re
from collections import Counter


BANNED_KEYWORDS = {
    "general", "discussion", "meeting", "topic", "content",
    "summary", "overview", "information", "points", "items",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "at", "for", "with", "by",
    "from", "as", "it", "its", "this", "that", "these", "those", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "can", "i", "we", "you", "they", "he", "she", "them",
    "us", "me", "my", "our", "your", "their", "his", "her", "so", "if",
    "than", "then", "there", "here", "about", "into", "out", "up", "down",
    "over", "under", "not", "no", "yes", "just", "like", "really", "very",
    "all", "some", "any", "each", "every", "one", "two", "three", "what",
    "when", "where", "which", "who", "why", "how", "also", "because",
    "while", "after", "before", "during", "between", "among", "again",
    "only", "own", "same", "more", "most", "other", "such", "too", "now",
    "still", "ever", "never", "well", "way", "thing", "things", "something",
    "someone", "anything", "anyone", "everything", "everyone",
}

# Last-ditch padding when neither the LLM nor the transcript yield anything
# usable. Chosen to describe the input rather than invent meeting topics, and
# guaranteed not to be in BANNED_KEYWORDS.
FINAL_FALLBACKS = ("audio", "segment", "window")


def extract_keywords_from_transcript(transcript: str, need: int) -> list[str]:
    """Return up to ``need`` keywords extracted from ``transcript``.

    Ranks tokens by frequency, breaks ties toward capitalized (proper-noun-like)
    tokens. Filters stopwords and banned terms. Returns an empty list when the
    transcript yields nothing usable.
    """
    if not transcript or need <= 0:
        return []
    tokens = re.findall(r"[A-Za-z][A-Za-z'\-]+", transcript)
    candidates = [
        t for t in tokens
        if len(t) > 2
        and t.lower() not in STOPWORDS
        and t.lower() not in BANNED_KEYWORDS
    ]
    if not candidates:
        return []
    counts = Counter(c.lower() for c in candidates)
    canonical: dict[str, str] = {}
    cap_keys: set[str] = set()
    for t in candidates:
        key = t.lower()
        if key not in canonical or (t[0].isupper() and not canonical[key][0].isupper()):
            canonical[key] = t
        if t[0].isupper():
            cap_keys.add(key)
    ranked = sorted(
        counts.keys(),
        key=lambda k: (-counts[k], 0 if k in cap_keys else 1, k),
    )
    return [canonical[k] for k in ranked[:need]]


def ensure_three_keywords(keywords: list, transcript: str = "") -> list[str]:
    """Guarantee exactly 3 keywords: dedupe, strip banned, backfill from transcript.

    Steps:
      1. Stringify, strip empties, drop banned terms.
      2. Dedupe case-insensitively preserving first-seen casing.
      3. If short of 3, pull candidates from the transcript.
      4. If still short (empty/meaningless transcript), pad from FINAL_FALLBACKS.
    """
    kw = [str(k).strip() for k in keywords if str(k).strip()]
    kw = [k for k in kw if k.lower() not in BANNED_KEYWORDS]
    seen: set[str] = set()
    deduped: list[str] = []
    for k in kw:
        key = k.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(k)
    kw = deduped
    if len(kw) < 3:
        extras = extract_keywords_from_transcript(transcript, 3 - len(kw) + 5)
        for e in extras:
            if e.lower() in seen or e.lower() in BANNED_KEYWORDS:
                continue
            kw.append(e)
            seen.add(e.lower())
            if len(kw) >= 3:
                break
    for fb in FINAL_FALLBACKS:
        if len(kw) >= 3:
            break
        if fb.lower() not in seen:
            kw.append(fb)
            seen.add(fb.lower())
    return kw[:3]
