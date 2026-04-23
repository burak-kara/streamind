import sys
import os
import json
import time
import types
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'plugins', 'nodes', 'proc', '_summarizer_mlx'))

# Stub mlx_lm + mlx_lm.sample_utils before import so tests run without MLX installed.
_mlx_lm_stub = types.ModuleType("mlx_lm")
_mlx_lm_stub.load = MagicMock()
_mlx_lm_stub.generate = MagicMock()
sys.modules.setdefault("mlx_lm", _mlx_lm_stub)
_mlx_sample_utils = types.ModuleType("mlx_lm.sample_utils")
_mlx_sample_utils.make_sampler = MagicMock(return_value=None)
_mlx_sample_utils.make_logits_processors = MagicMock(return_value=None)
sys.modules.setdefault("mlx_lm.sample_utils", _mlx_sample_utils)

from juturna.components import Message
from juturna.payloads import ObjectPayload

import summarizer_mlx  # noqa: E402
from summarizer_mlx import SummarizerMlx, BANNED_KEYWORDS  # noqa: E402


def _make_window_message(transcript: str, window_id: int = 0) -> Message[ObjectPayload]:
    payload = ObjectPayload.from_dict({
        "window_id": window_id,
        "window_start": window_id * 300.0,
        "window_end": (window_id + 1) * 300.0,
        "full_transcript": transcript,
        "trigger_time": time.time(),
    })
    return Message[ObjectPayload](creator="test", version=1, payload=payload)


def _make_node(tokenizer_mock=None) -> SummarizerMlx:
    node = SummarizerMlx(
        model_name="fake/model",
        prompt_template_file="summarize_prompt_mlx_qwen3.txt",
    )
    node._name = "test_summarizer"
    node.transmit = MagicMock()
    node._prompt_template = "Transcript:\n{transcript}"
    node._mlx_model = object()
    tok = tokenizer_mock or MagicMock()
    tok.apply_chat_template.return_value = "formatted prompt"
    node._mlx_tokenizer = tok
    return node


class TestSummarizerMlx:
    def test_produces_summary_and_keywords(self):
        node = _make_node()
        with patch.object(summarizer_mlx, "generate", return_value=json.dumps({
            "summary": "Alice briefed Bob on the API rollout.",
            "keywords": ["API", "rollout", "Alice"],
        })):
            node.update(_make_window_message("Alice briefed Bob about API rollout."))
        result = node.transmit.call_args[0][0].payload
        assert len(result["keywords"]) == 3
        assert result["summary"].startswith("Alice")
        assert isinstance(result["latency"], float)
        assert result["model_name"] == "fake/model"

    def test_malformed_response_pads_keywords(self):
        node = _make_node()
        with patch.object(summarizer_mlx, "generate", return_value=json.dumps({
            "summary": "Partial output",
            "keywords": ["rollout"],
        })):
            node.update(_make_window_message("Alice briefed Bob about the API rollout and timeline."))
        result = node.transmit.call_args[0][0].payload
        kws = result["keywords"]
        assert len(kws) == 3
        assert all(k.lower() not in BANNED_KEYWORDS for k in kws)

    def test_error_path_fills_keywords_from_transcript(self):
        node = _make_node()
        transcript = (
            "Alice briefed Bob on the API rollout timeline. "
            "Alice emphasized security review. Bob agreed on rollout scope."
        )
        with patch.object(summarizer_mlx, "generate", side_effect=RuntimeError("boom")):
            node.update(_make_window_message(transcript))
        result = node.transmit.call_args[0][0].payload
        kws = result["keywords"]
        assert result["summary"] == ""
        assert len(kws) == 3
        assert all(k.lower() not in BANNED_KEYWORDS for k in kws)
        # Must draw from transcript content, not the old "general" placeholder.
        assert all(k.lower() != "general" for k in kws)
        assert any(k.lower() in transcript.lower() for k in kws)

    def test_trailing_eot_token_stripped(self):
        node = _make_node()
        raw = json.dumps({
            "summary": "Alice briefed Bob on API rollout.",
            "keywords": ["API", "rollout", "Alice"],
        }) + "<|im_end|>"
        with patch.object(summarizer_mlx, "generate", return_value=raw):
            node.update(_make_window_message("Alice briefed Bob about API rollout."))
        result = node.transmit.call_args[0][0].payload
        assert result["summary"].startswith("Alice")
        assert len(result["keywords"]) == 3

    def test_trailing_commentary_after_json_tolerated(self):
        node = _make_node()
        raw = json.dumps({
            "summary": "Alice briefed Bob.",
            "keywords": ["Alice", "Bob", "brief"],
        }) + "\n\nNote: this is a follow-up comment from the model."
        with patch.object(summarizer_mlx, "generate", return_value=raw):
            node.update(_make_window_message("Alice briefed Bob."))
        result = node.transmit.call_args[0][0].payload
        assert result["summary"] == "Alice briefed Bob."
        assert len(result["keywords"]) == 3

    def test_banned_keywords_stripped(self):
        node = _make_node()
        transcript = "Alice briefed Bob on the API rollout timeline and security review scope."
        with patch.object(summarizer_mlx, "generate", return_value=json.dumps({
            "summary": "ok",
            "keywords": ["general", "meeting", "API"],
        })):
            node.update(_make_window_message(transcript))
        result = node.transmit.call_args[0][0].payload
        kws = [k.lower() for k in result["keywords"]]
        assert "general" not in kws
        assert "meeting" not in kws
        assert "api" in kws
        assert len(result["keywords"]) == 3


