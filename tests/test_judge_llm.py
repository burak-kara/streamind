import importlib.util
import json
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "plugins" / "nodes" / "sink" / "_judge_llm"))

from juturna.components import Message
from juturna.payloads import ObjectPayload


def _load_scorer():
    spec = importlib.util.spec_from_file_location(
        "judge_scorer_test",
        REPO_ROOT / "plugins" / "nodes" / "sink" / "_judge_common" / "scorer.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["judge_scorer_test"] = mod
    spec.loader.exec_module(mod)
    return mod


_scorer = _load_scorer()


def _make_summary_message(window_id: int = 0, transcript: str = "Some transcript text.",
                           summary: str = "A summary.",
                           keywords: list[str] | None = None,
                           latency: float = 1.5,
                           summarizer_model: str = "qwen3.5:4b") -> Message[ObjectPayload]:
    payload = ObjectPayload.from_dict({
        "window_id": window_id,
        "window_start": window_id * 300.0,
        "window_end": (window_id + 1) * 300.0,
        "summary": summary,
        "keywords": keywords if keywords is not None else ["alpha", "beta", "gamma"],
        "latency": latency,
        "model_name": summarizer_model,
        "full_transcript": transcript,
    })
    return Message[ObjectPayload](creator="test", version=1, payload=payload)


def _good_judge_response(b_values=(4, 4, 4, 4, 4), flags=(True, True, True)) -> str:
    return json.dumps({
        "factual_consistency": b_values[0],
        "relevance": b_values[1],
        "coherence": b_values[2],
        "fluency": b_values[3],
        "conciseness": b_values[4],
        "keyword_relevance": list(flags),
    })


class TestScorer:

    def test_compute_score_applies_latency_gate(self):
        # B = 5*4 = 20, K = +6, fast latency -> L close to 6
        b = {c: 4 for c in _scorer.LIKERT_CRITERIA}
        score = _scorer.compute_score(b, [True, True, True], proc_time=1.0)
        assert score.b == 20
        assert score.k == 6
        # 10 * exp(-0.5) ~= 6.065
        assert 6.0 < score.l < 6.1
        assert score.c == pytest.approx(score.b + score.k + score.l)

    def test_compute_score_zeroes_latency_when_b_below_10(self):
        # B = 5*1 = 5, below the gate
        b = {c: 1 for c in _scorer.LIKERT_CRITERIA}
        score = _scorer.compute_score(b, [True, True, True], proc_time=1.0)
        assert score.b == 5
        assert score.l == 0.0
        assert score.c == 5 + 6 + 0.0

    def test_keyword_score_handles_mixed_flags(self):
        b = {c: 3 for c in _scorer.LIKERT_CRITERIA}
        score = _scorer.compute_score(b, [True, False, True], proc_time=2.0)
        assert score.k == 2  # +2 -2 +2

    def test_parse_response_strips_fences_and_commentary(self):
        raw = "```json\n" + _good_judge_response() + "\n```\nblah blah after"
        b, flags = _scorer.parse_judge_response(raw)
        assert sum(b.values()) == 20
        assert flags == [True, True, True]

    def test_parse_response_clamps_likert_outliers(self):
        raw = json.dumps({
            "factual_consistency": 9,
            "relevance": -1,
            "coherence": 0,
            "fluency": 5,
            "conciseness": 3,
            "keyword_relevance": [True, False, True],
        })
        b, flags = _scorer.parse_judge_response(raw)
        # 9 clamped to 5; non-positive values treated as missing (0) so a
        # malformed judge response doesn't actively penalise the candidate.
        assert b["factual_consistency"] == 5
        assert b["relevance"] == 0
        assert b["coherence"] == 0
        assert flags == [True, False, True]

    def test_parse_response_pads_missing_keyword_flags(self):
        raw = json.dumps({
            "factual_consistency": 4,
            "relevance": 4,
            "coherence": 4,
            "fluency": 4,
            "conciseness": 4,
            "keyword_relevance": [True],
        })
        _, flags = _scorer.parse_judge_response(raw)
        assert flags == [True, False, False]

    def test_parse_response_raises_on_no_json(self):
        with pytest.raises(ValueError):
            _scorer.parse_judge_response("totally not JSON")


class TestJudgeLLMNode:

    def _make_node(self, results_dir: str, queue_size: int = 8):
        from judge_llm import JudgeLLM
        node = JudgeLLM(
            endpoint="http://localhost:11434",
            model_name="qwen3.5:4b",
            results_dir=results_dir,
            queue_size=queue_size,
        )
        node._name = "test_judge"
        return node

    @patch("judge_llm.ollama.Client")
    def test_writes_judge_record_to_disk(self, mock_client_cls):
        mock_client = MagicMock()
        resp = MagicMock()
        resp.message.content = _good_judge_response()
        mock_client.chat.return_value = resp
        mock_client_cls.return_value = mock_client

        with tempfile.TemporaryDirectory() as tmpdir:
            node = self._make_node(tmpdir)
            node.warmup()
            try:
                msg = _make_summary_message(window_id=2, latency=1.0)
                node.update(msg)
                # Wait for the worker to finish processing the single item.
                deadline = time.time() + 5.0
                expected = Path(tmpdir) / "qwen3.5-4b" / "300" / "judge" / "window_2.json"
                while time.time() < deadline and not expected.exists():
                    time.sleep(0.05)
                assert expected.exists(), "judge record was not written"
                data = json.loads(expected.read_text())
                # Challenge-shaped fields preserved + judge fields added.
                assert data["from"] == 600.0
                assert data["to"] == 900.0
                assert data["proc_time"] == 1.0
                assert data["judge_model"] == "qwen3.5:4b"
                assert data["B"] == 20
                assert data["K"] == 6
                assert "B_breakdown" in data
                # B >= 10 so L > 0
                assert data["L"] > 0
                assert data["C"] == pytest.approx(data["B"] + data["K"] + data["L"])
            finally:
                node.stop()

    @patch("judge_llm.ollama.Client")
    def test_malformed_judge_json_does_not_crash(self, mock_client_cls):
        mock_client = MagicMock()
        resp = MagicMock()
        resp.message.content = "the model spoke prose instead of JSON"
        mock_client.chat.return_value = resp
        mock_client_cls.return_value = mock_client

        with tempfile.TemporaryDirectory() as tmpdir:
            node = self._make_node(tmpdir)
            node.warmup()
            try:
                node.update(_make_summary_message(window_id=0))
                # Give the worker a moment to fail and keep running.
                time.sleep(0.5)
                # Worker should still be alive — failure logged, not raised.
                assert node._worker.is_alive()
                # And no record should be on disk.
                judge_dir = Path(tmpdir) / "qwen3.5-4b" / "300" / "judge"
                if judge_dir.exists():
                    assert not any(judge_dir.iterdir())
            finally:
                node.stop()

    @patch("judge_llm.ollama.Client")
    def test_drop_oldest_when_queue_full(self, mock_client_cls):
        # Make Ollama "hang" so the worker can't drain the queue.
        block_event = threading.Event()

        def slow_chat(*args, **kwargs):
            block_event.wait(timeout=5.0)
            r = MagicMock()
            r.message.content = _good_judge_response()
            return r

        mock_client = MagicMock()
        mock_client.chat.side_effect = slow_chat
        mock_client_cls.return_value = mock_client

        with tempfile.TemporaryDirectory() as tmpdir:
            node = self._make_node(tmpdir, queue_size=2)
            node.warmup()
            try:
                # First item gets pulled by the worker (it's blocked in chat).
                # Then queue can hold 2 more before being full.
                for i in range(6):
                    node.update(_make_summary_message(window_id=i))
                # Queue should be at most queue_size deep (2). The worker is
                # holding one item in flight, the queue holds the most recent two.
                assert node._work_queue.qsize() <= 2
            finally:
                block_event.set()
                node.stop()

    @patch("judge_llm.ollama.Client")
    def test_update_returns_immediately(self, mock_client_cls):
        # If the live path were ever blocked on the judge it would defeat the
        # whole point of an offline node. update() must be O(1) on a fast queue.
        block_event = threading.Event()

        def slow_chat(*args, **kwargs):
            block_event.wait(timeout=5.0)
            r = MagicMock()
            r.message.content = _good_judge_response()
            return r

        mock_client = MagicMock()
        mock_client.chat.side_effect = slow_chat
        mock_client_cls.return_value = mock_client

        with tempfile.TemporaryDirectory() as tmpdir:
            node = self._make_node(tmpdir, queue_size=4)
            node.warmup()
            try:
                start = time.time()
                node.update(_make_summary_message(window_id=0))
                elapsed = time.time() - start
                assert elapsed < 0.1, f"update() took {elapsed:.3f}s — must not block"
            finally:
                block_event.set()
                node.stop()
