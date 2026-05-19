"""Tests for the shared keyword-post-processing module used by the
vLLM summarizer node.
"""

import importlib.util
import os
import sys

import pytest


_SHARED_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "plugins",
    "nodes",
    "proc",
    "_summarizer_common",
    "keywords.py",
)
_spec = importlib.util.spec_from_file_location("summarizer_keywords", _SHARED_PATH)
keywords_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(keywords_mod)

BANNED_KEYWORDS = keywords_mod.BANNED_KEYWORDS
FINAL_FALLBACKS = keywords_mod.FINAL_FALLBACKS
extract_keywords_from_transcript = keywords_mod.extract_keywords_from_transcript
ensure_three_keywords = keywords_mod.ensure_three_keywords


class TestExtractKeywordsFromTranscript:
    def test_empty_transcript(self):
        assert extract_keywords_from_transcript("", 3) == []

    def test_zero_need(self):
        assert extract_keywords_from_transcript("anything here", 0) == []

    def test_prefers_capitalized_tokens(self):
        txt = "Alice met Bob. Alice discussed strategy. strategy aligned with Alice."
        out = extract_keywords_from_transcript(txt, 3)
        assert "Alice" in out

    def test_filters_banned_and_stopwords(self):
        txt = "This is a general meeting about the discussion of things."
        out = extract_keywords_from_transcript(txt, 3)
        assert all(w.lower() not in BANNED_KEYWORDS for w in out)


class TestEnsureThreeKeywords:
    def test_passthrough_exact_three(self):
        out = ensure_three_keywords(["alpha", "beta", "gamma"])
        assert out == ["alpha", "beta", "gamma"]

    def test_strips_banned_keywords(self):
        out = ensure_three_keywords(
            ["general", "meeting", "API"],
            transcript="Alice briefed Bob on API rollout timeline."
        )
        lowered = [k.lower() for k in out]
        assert "general" not in lowered
        assert "meeting" not in lowered
        assert "api" in lowered
        assert len(out) == 3

    def test_dedupes_case_insensitive(self):
        out = ensure_three_keywords(
            ["Alpha", "alpha", "Beta"],
            transcript="gamma delta epsilon",
        )
        assert len(out) == 3
        lowered = [k.lower() for k in out]
        assert lowered.count("alpha") == 1

    def test_backfills_from_transcript(self):
        out = ensure_three_keywords(
            [],
            transcript="Alice briefed Bob on the API rollout timeline.",
        )
        assert len(out) == 3
        assert all(k.lower() not in BANNED_KEYWORDS for k in out)

    def test_static_fallback_when_transcript_empty(self):
        out = ensure_three_keywords([], transcript="")
        assert len(out) == 3
        assert out == list(FINAL_FALLBACKS)

    def test_truncates_when_over_three(self):
        out = ensure_three_keywords(
            ["a", "b", "c", "d", "e"],
            transcript="",
        )
        assert out == ["a", "b", "c"]

    def test_drops_empty_and_whitespace(self):
        out = ensure_three_keywords(
            ["", "   ", "Alpha", "Beta"],
            transcript="Gamma Delta Epsilon",
        )
        assert len(out) == 3
        assert "" not in out
