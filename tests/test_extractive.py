"""Unit tests for the deterministic extractive fallback summary."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_EX_PATH = _HERE.parent / "plugins" / "nodes" / "proc" / "_summarizer_common" / "extractive.py"
_spec = importlib.util.spec_from_file_location("extractive_test", _EX_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["extractive_test"] = _mod
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
extractive_summary = _mod.extractive_summary


def test_empty_returns_marker():
    assert extractive_summary("") == "[no transcript content]"
    assert extractive_summary("   \n\t ") == "[no transcript content]"


def test_short_input_returned_verbatim():
    s = "Just a short sentence."
    assert extractive_summary(s, max_chars=300) == s


def test_long_input_cuts_at_sentence_boundary():
    text = (
        "First clear sentence ends here. "
        "Second sentence with more content goes after. "
        "Third one as well."
    )
    out = extractive_summary(text, max_chars=80)
    # Must end at a sentence boundary, not mid-word.
    assert out.endswith(".")
    assert "First clear sentence ends here." in out
    assert len(out) <= 80


def test_no_sentence_boundary_falls_back_to_word():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    out = extractive_summary(text, max_chars=20)
    assert out.endswith("…")
    # No mid-word cut.
    assert " " not in out[-5:-1] or out.rstrip("…").endswith(("alpha", "beta", "gamma", "delta"))
    assert len(out) <= 21  # 20 + ellipsis


def test_question_and_exclamation_boundaries():
    text = "Why did this happen? Because the config was wrong! Now we fix it. More words."
    out = extractive_summary(text, max_chars=60)
    assert out.endswith(("?", "!", "."))


def test_strips_trailing_whitespace_in_short_path():
    assert extractive_summary("   hello world   ") == "hello world"


def test_refuses_one_word_summary():
    # Sentence boundary at position 6 would yield "Hi." — too short. The function
    # demands the boundary be past half of max_chars, so it should fall through.
    text = "Hi. " + "x" * 200
    out = extractive_summary(text, max_chars=100)
    # The dot at position 2 is well under half of 100 → boundary rejected →
    # word-boundary path emits "…"
    assert out.endswith("…") or len(out) > 50
