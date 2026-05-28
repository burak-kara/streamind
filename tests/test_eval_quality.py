"""Tests for the offline judge harness in tools/eval_quality.py.

vLLM is mocked via sys.modules injection so tests run on any platform.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _install_fake_vllm(judge_text: str):
    fake = types.ModuleType("vllm")

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Out:
        def __init__(self, text):
            self.outputs = [types.SimpleNamespace(text=text)]

    class LLM:
        def __init__(self, *args, **kwargs):
            self.init_kwargs = kwargs

        def chat(self, messages, sampling_params=None, use_tqdm=False):
            return [_Out(judge_text)]

    fake.LLM = LLM
    fake.SamplingParams = SamplingParams
    sys.modules["vllm"] = fake


def _load_eval_quality():
    if "eval_quality" in sys.modules:
        del sys.modules["eval_quality"]
    spec = importlib.util.spec_from_file_location(
        "eval_quality", REPO_ROOT / "tools" / "eval_quality.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_quality"] = mod
    spec.loader.exec_module(mod)
    return mod


def _good_judge_response(b=(4, 4, 4, 4, 4), flags=(True, True, True)) -> str:
    return json.dumps({
        "factual_consistency": b[0],
        "relevance": b[1],
        "coherence": b[2],
        "fluency": b[3],
        "conciseness": b[4],
        "keyword_relevance": list(flags),
    })


def _populate_results(tmp_path: Path, n_windows: int = 2) -> Path:
    results_root = tmp_path / "results" / "fake-model" / "300"
    results_root.mkdir(parents=True)
    debug_dir = results_root / "debug"
    debug_dir.mkdir()
    for i in range(n_windows):
        (results_root / f"window_{i}.json").write_text(json.dumps({
            "from": float(i * 300),
            "to": float((i + 1) * 300),
            "summary": f"Summary {i}.",
            "keywords": ["alpha", "beta", "gamma"],
            "proc_time": 1.5,
        }))
        (debug_dir / f"window_{i}_transcript.txt").write_text(
            f"Transcript text for window {i}."
        )
    return results_root


def _write_judge_profile(tmp_path: Path, model_dir: Path, name: str = "vllm-test-judge"):
    pipelines_dir = tmp_path / "pipelines" / "judge"
    pipelines_dir.mkdir(parents=True)
    (pipelines_dir / f"{name}.json").write_text(json.dumps({
        "name": "judge",
        "type": "sink",
        "mark": "judge_vllm",
        "configuration": {
            "model_name": str(model_dir),
            "dtype": "float16",
            "gpu_memory_utilization": 0.85,
            "max_model_len": 4096,
            "max_tokens": 256,
            "temperature": 0.0,
        },
    }))
    return name


def _populate_fake_model(tmp_path: Path) -> Path:
    model_dir = tmp_path / "models" / "fake-judge"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model_type": "qwen2"}')
    return model_dir


def test_score_one_emits_judge_per_window(tmp_path, monkeypatch):
    _install_fake_vllm(_good_judge_response())
    results_root = _populate_results(tmp_path, n_windows=2)
    model_dir = _populate_fake_model(tmp_path)

    monkeypatch.chdir(tmp_path)
    profile_name = _write_judge_profile(tmp_path, model_dir)
    # The script resolves _REPO_ROOT relative to its own __file__, so we need
    # to monkeypatch the loaded module's pipelines/judge lookup.
    eq = _load_eval_quality()
    monkeypatch.setattr(eq, "_REPO_ROOT", tmp_path)

    cfg = eq._load_judge_profile(profile_name)
    llm, sampling = eq._build_llm(cfg)
    windows = list(eq._iter_window_files(results_root))
    assert len(windows) == 2

    scored = [eq._score_one(llm, sampling, w) for w in windows]
    scored = [s for s in scored if s is not None]
    assert len(scored) == 2
    for s in scored:
        assert s.b == 20  # 5 * 4
        assert s.k == 6
        assert s.c == pytest.approx(s.b + s.k + s.l)

    eq._write_per_window(scored, judge_model=cfg["model_name"])

    judge_dir = results_root / "judge"
    assert judge_dir.is_dir()
    for i in range(2):
        out = json.loads((judge_dir / f"window_{i}.json").read_text())
        # Original keys preserved
        assert "from" in out and "to" in out and "summary" in out
        # B/K/L/C present
        assert out["B"] == 20
        assert out["K"] == 6
        assert "L" in out and "C" in out
        assert out["judge_model"] == str(model_dir)


def test_iter_window_files_skips_debug_and_judge_dirs(tmp_path):
    results_root = _populate_results(tmp_path, n_windows=1)
    # Add a noise window inside judge/ — must be skipped
    judge_dir = results_root / "judge"
    judge_dir.mkdir()
    (judge_dir / "window_0.json").write_text("{}")

    eq = _load_eval_quality()
    paths = list(eq._iter_window_files(results_root))
    assert len(paths) == 1
    assert "judge" not in paths[0].parts


def test_build_llm_rejects_missing_model_dir(tmp_path, monkeypatch):
    _install_fake_vllm(_good_judge_response())
    eq = _load_eval_quality()
    cfg = {"model_name": str(tmp_path / "missing")}
    with pytest.raises(FileNotFoundError) as exc:
        eq._build_llm(cfg)
    assert "fetch_models.sh" in str(exc.value)


def test_load_judge_profile_missing_raises(tmp_path, monkeypatch):
    eq = _load_eval_quality()
    monkeypatch.setattr(eq, "_REPO_ROOT", tmp_path)
    (tmp_path / "pipelines" / "judge").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        eq._load_judge_profile("does-not-exist")


# --- sliding-window WER -----------------------------------------------------


def _maybe_skip_no_jiwer():
    try:
        import jiwer  # noqa: F401
    except ImportError:
        pytest.skip("jiwer not installed; WER tests need the dev extra.")


def test_sliding_wer_returns_min_over_offsets():
    _maybe_skip_no_jiwer()
    eq = _load_eval_quality()
    # Reference is two distinct halves; "correct" alignment lands in the
    # second half. Proportional anchor picks the boundary; sliding should
    # find the second-half slice and report low WER.
    ref = (
        "alpha bravo charlie delta echo foxtrot " * 5
        + "golf hotel india juliet kilo lima " * 5
    )
    total_dur = 60.0
    # Hypothesis matches the second half verbatim.
    hyp = "golf hotel india juliet kilo lima " * 5
    # Window 30-60s sits exactly on the second half; anchor 0 is fine.
    wer_anchor, _ = eq._sliding_wer(ref, 30.0, 60.0, total_dur, hyp)
    assert wer_anchor is not None
    assert wer_anchor < 0.05

    # Now shift the window so the anchor is wrong; sliding must rescue it.
    # Window 22-52s anchors into the first half; +offsets push into second.
    wer_misaligned, slice_text = eq._sliding_wer(ref, 22.0, 52.0, total_dur, hyp)
    wer_naive = eq._compute_wer(
        eq._slice_proportional(ref, 22.0, 52.0, total_dur), hyp
    )
    assert wer_misaligned is not None and wer_naive is not None
    assert wer_misaligned <= wer_naive
    assert "golf" in slice_text or "hotel" in slice_text


def test_sliding_wer_handles_empty_inputs():
    eq = _load_eval_quality()
    assert eq._sliding_wer("", 0.0, 10.0, 100.0, "anything") == (None, "")
    assert eq._sliding_wer("ref", 0.0, 10.0, 0.0, "anything") == (None, "")
    assert eq._sliding_wer("ref", 0.0, 10.0, 100.0, "") == (None, "")
    assert eq._sliding_wer("ref", 0.0, 10.0, 100.0, "  ") == (None, "")
