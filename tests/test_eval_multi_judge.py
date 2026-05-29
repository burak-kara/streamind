"""Tests for the multi-judge aggregation in tools/eval_multi_judge.py.

aggregate_scores is a pure function over per-judge json-out dicts, so these
tests run on any platform without vLLM or a GPU.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    if "eval_multi_judge" in sys.modules:
        del sys.modules["eval_multi_judge"]
    spec = importlib.util.spec_from_file_location(
        "eval_multi_judge", REPO_ROOT / "tools" / "eval_multi_judge.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_multi_judge"] = mod
    spec.loader.exec_module(mod)
    return mod


def _row(path, c, b=20, k=6, l=6.0):
    return {"path": path, "B": b, "K": k, "L": l, "C": c}


@pytest.fixture
def mod():
    return _load_module()


def test_mean_and_stdev_per_window(mod):
    per_judge = {
        "judgeA": [_row("w0", 30.0), _row("w1", 20.0)],
        "judgeB": [_row("w0", 34.0), _row("w1", 24.0)],
    }
    agg = mod.aggregate_scores(per_judge, janus_bonus=4.0)

    w0 = agg["windows"][0]
    assert w0["path"] == "w0"
    assert w0["mean_c"] == pytest.approx(32.0)
    assert w0["stdev_c"] == pytest.approx(2.0)  # pstdev of [30, 34]
    assert w0["per_judge"]["judgeA"]["C"] == 30.0
    assert w0["per_judge"]["judgeB"]["C"] == 34.0


def test_per_judge_final_source_score(mod):
    per_judge = {
        "judgeA": [_row("w0", 30.0), _row("w1", 20.0)],  # avg 25
        "judgeB": [_row("w0", 34.0), _row("w1", 24.0)],  # avg 29
    }
    agg = mod.aggregate_scores(per_judge, janus_bonus=4.0)

    assert agg["judge_summary"]["judgeA"]["avg_c"] == pytest.approx(25.0)
    assert agg["judge_summary"]["judgeA"]["final_source_score"] == pytest.approx(29.0)
    assert agg["judge_summary"]["judgeB"]["final_source_score"] == pytest.approx(33.0)


def test_consensus_score(mod):
    per_judge = {
        "judgeA": [_row("w0", 30.0), _row("w1", 20.0)],
        "judgeB": [_row("w0", 34.0), _row("w1", 24.0)],
    }
    agg = mod.aggregate_scores(per_judge, janus_bonus=4.0)
    # window means: (32, 22) -> consensus avg 27 -> +4 = 31
    assert agg["consensus"]["avg_c"] == pytest.approx(27.0)
    assert agg["consensus"]["final_source_score"] == pytest.approx(31.0)
    assert agg["consensus"]["n_windows"] == 2


def test_no_janus_bonus(mod):
    per_judge = {"judgeA": [_row("w0", 30.0)]}
    agg = mod.aggregate_scores(per_judge, janus_bonus=0.0)
    assert agg["judge_summary"]["judgeA"]["final_source_score"] == pytest.approx(30.0)
    assert agg["consensus"]["final_source_score"] == pytest.approx(30.0)


def test_mismatched_window_sets(mod):
    # judgeB skipped w1 (parse failure); stats over available judges only.
    per_judge = {
        "judgeA": [_row("w0", 30.0), _row("w1", 20.0)],
        "judgeB": [_row("w0", 34.0)],
    }
    agg = mod.aggregate_scores(per_judge, janus_bonus=4.0)

    paths = [w["path"] for w in agg["windows"]]
    assert paths == ["w0", "w1"]  # union, first-seen order

    w1 = agg["windows"][1]
    assert w1["mean_c"] == pytest.approx(20.0)  # only judgeA
    assert w1["stdev_c"] == 0.0  # single value
    assert "judgeB" not in w1["per_judge"]


def test_single_judge_stdev_zero(mod):
    per_judge = {"judgeA": [_row("w0", 30.0), _row("w1", 20.0)]}
    agg = mod.aggregate_scores(per_judge, janus_bonus=4.0)
    assert all(w["stdev_c"] == 0.0 for w in agg["windows"])
    assert agg["judges"] == ["judgeA"]


def test_build_table_runs(mod):
    per_judge = {
        "judgeA": [_row("w0", 30.0), _row("w1", 20.0)],
        "judgeB": [_row("w0", 34.0), _row("w1", 24.0)],
    }
    agg = mod.aggregate_scores(per_judge, janus_bonus=4.0)
    table = mod.build_table(agg, janus_bonus=4.0)
    assert "final source score" in table
    assert "Consensus" in table
    assert "judgeA" in table or "judgeA"[-10:] in table
