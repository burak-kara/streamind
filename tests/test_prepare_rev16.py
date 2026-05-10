"""Smoke tests for tools/finetune/prepare_rev16.py.

Tests the pure helpers (no Ollama / no GPU required) to make sure the
distillation pipeline's plumbing — windowing, JSON parsing, episode-level
splitting — keeps working as the script evolves.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_rev16",
        REPO_ROOT / "tools" / "finetune" / "prepare_rev16.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prepare_rev16"] = mod
    spec.loader.exec_module(mod)
    return mod


prep = _load_module()


class TestWindowing:

    def test_short_transcript_yields_single_window(self):
        text = " ".join(["word"] * 800)
        windows = prep._split_into_windows(text, window_words=600, stride_words=None)
        assert len(windows) == 1
        assert windows[0].count(" ") == 599  # 600 words → 599 spaces

    def test_long_transcript_yields_multiple_windows_no_overlap(self):
        text = " ".join(f"w{i}" for i in range(2000))
        windows = prep._split_into_windows(text, window_words=600, stride_words=None)
        # 2000 / 600 = 3 full windows; trailing 200 < 300 (half window) → drop
        assert len(windows) == 3
        for w in windows:
            assert len(w.split()) == 600
        # Verify no overlap: word indices in the first 3 windows are disjoint.
        first_words = windows[0].split()
        second_words = windows[1].split()
        assert first_words[-1] != second_words[0]

    def test_overlapping_stride(self):
        text = " ".join(f"w{i}" for i in range(1500))
        windows = prep._split_into_windows(text, window_words=600, stride_words=300)
        # Stride 300: starts at 0, 300, 600, 900 (each full 600-word window),
        # plus 1200 (300 words — exactly half, retained).
        assert len(windows) == 5
        assert windows[0].split()[0] == "w0"
        assert windows[1].split()[0] == "w300"
        # The trailing partial window is shorter but still a valid example.
        assert len(windows[-1].split()) == 300

    def test_empty_transcript_yields_no_windows(self):
        assert prep._split_into_windows("", 600, None) == []
        assert prep._split_into_windows("   ", 600, None) == []


class TestParseSummary:

    def test_strips_code_fences(self):
        raw = '```json\n{"summary": "ok", "keywords": ["a","b","c"]}\n```'
        s, kw = prep._parse_summary_response(raw)
        assert s == "ok"
        assert kw == ["a", "b", "c"]

    def test_strips_qwen_special_tokens(self):
        raw = '<|im_start|>{"summary": "ok", "keywords": ["a","b","c"]}<|im_end|>'
        s, kw = prep._parse_summary_response(raw)
        assert s == "ok"
        assert kw == ["a", "b", "c"]

    def test_drops_empty_keywords(self):
        raw = '{"summary": "ok", "keywords": ["a", "", " ", "c"]}'
        s, kw = prep._parse_summary_response(raw)
        assert kw == ["a", "c"]

    def test_raises_on_no_json(self):
        with pytest.raises(ValueError):
            prep._parse_summary_response("totally not JSON")


class TestEpisodeSplit:

    def test_no_overlap_between_splits(self):
        eps = [f"ep{i}" for i in range(16)]
        train, val, test = prep._split_episodes(eps, val_frac=0.1, test_frac=0.125, seed=42)
        # Disjoint
        assert set(train).isdisjoint(set(val))
        assert set(train).isdisjoint(set(test))
        assert set(val).isdisjoint(set(test))
        # Coverage
        assert set(train) | set(val) | set(test) == set(eps)
        # Sizes — 16 * 0.125 = 2 test, 16 * 0.1 ≈ 1.6 → 2 val (round)
        assert len(test) == 2
        assert len(val) == 2
        assert len(train) == 12

    def test_tiny_corpus_degrades_gracefully(self):
        eps = ["a", "b", "c"]
        train, val, test = prep._split_episodes(eps, val_frac=0.1, test_frac=0.125, seed=0)
        # Should not produce empty train.
        assert len(train) >= 1
        assert len(test) == 1
        assert len(val) == 1
        assert set(train) | set(val) | set(test) == set(eps)

    def test_zero_frac_returns_empty(self):
        eps = [f"ep{i}" for i in range(10)]
        train, val, test = prep._split_episodes(eps, val_frac=0.0, test_frac=0.0, seed=1)
        assert val == []
        assert test == []
        assert sorted(train) == sorted(eps)
