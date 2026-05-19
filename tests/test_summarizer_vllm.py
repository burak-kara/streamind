"""Tests for the vLLM in-process summarizer node.

vLLM only installs on Linux+CUDA, so these tests inject a fake `vllm`
module via sys.modules before the node imports it. Run anywhere — no GPU
needed.
"""
from __future__ import annotations

import importlib
import json
import sys
import time
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NODE_DIR = REPO_ROOT / "plugins" / "nodes" / "proc" / "_summarizer_vllm"
sys.path.insert(0, str(NODE_DIR))

from juturna.components import Message
from juturna.payloads import ObjectPayload


def _install_fake_vllm(generate_text: str):
    """Inject a fake `vllm` module so `from vllm import LLM, SamplingParams` succeeds."""
    fake = types.ModuleType("vllm")

    sampling_calls = []

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            sampling_calls.append(kwargs)

    class _Out:
        def __init__(self, text):
            self.outputs = [types.SimpleNamespace(text=text)]

    class LLM:
        def __init__(self, *args, **kwargs):
            self.init_kwargs = kwargs

        def chat(self, messages, sampling_params=None, use_tqdm=False):
            return [_Out(generate_text)]

    fake.LLM = LLM
    fake.SamplingParams = SamplingParams
    sys.modules["vllm"] = fake
    return fake, sampling_calls


def _populate_fake_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "models" / "fake"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model_type": "qwen2"}')
    return model_dir


def _make_window_message(transcript: str, window_id: int = 0) -> Message[ObjectPayload]:
    payload = ObjectPayload.from_dict({
        "window_id": window_id,
        "window_start": window_id * 300.0,
        "window_end": (window_id + 1) * 300.0,
        "full_transcript": transcript,
        "trigger_time": time.time(),
    })
    return Message[ObjectPayload](creator="test", version=1, payload=payload)


@pytest.fixture
def reload_node():
    """Force a fresh import of summarizer_vllm per test so module-level state stays clean."""
    if "summarizer_vllm" in sys.modules:
        del sys.modules["summarizer_vllm"]
    yield
    if "summarizer_vllm" in sys.modules:
        del sys.modules["summarizer_vllm"]


def test_produces_summary_and_three_keywords(tmp_path, reload_node):
    _install_fake_vllm(json.dumps({
        "summary": "The team discussed roadmap milestones.",
        "keywords": ["roadmap", "milestones", "team"],
    }))
    model_dir = _populate_fake_model_dir(tmp_path)

    from summarizer_vllm import SummarizerVLLM

    node = SummarizerVLLM(model_name=str(model_dir))
    node._name = "test"
    node.transmit = MagicMock()
    node.warmup()
    node.update(_make_window_message("We covered the Q3 roadmap."))

    assert node.transmit.call_count == 1
    out = node.transmit.call_args[0][0].payload
    assert out["summary"] == "The team discussed roadmap milestones."
    assert len(out["keywords"]) == 3
    assert isinstance(out["latency"], float)
    assert out["model_name"] == str(model_dir)


def test_malformed_response_pads_keywords(tmp_path, reload_node):
    _install_fake_vllm(json.dumps({
        "summary": "Some summary.",
        "keywords": ["only_one"],
    }))
    model_dir = _populate_fake_model_dir(tmp_path)

    from summarizer_vllm import SummarizerVLLM

    node = SummarizerVLLM(model_name=str(model_dir))
    node._name = "test"
    node.transmit = MagicMock()
    node.warmup()
    node.update(_make_window_message("Some meeting content."))

    out = node.transmit.call_args[0][0].payload
    assert len(out["keywords"]) == 3


def test_non_json_output_recovers_empty_summary(tmp_path, reload_node):
    _install_fake_vllm("totally not JSON, just chatty text from a confused model")
    model_dir = _populate_fake_model_dir(tmp_path)

    from summarizer_vllm import SummarizerVLLM

    node = SummarizerVLLM(model_name=str(model_dir))
    node._name = "test"
    node.transmit = MagicMock()
    node.warmup()
    node.update(_make_window_message("Transcript text."))

    out = node.transmit.call_args[0][0].payload
    assert out["summary"] == ""
    assert len(out["keywords"]) == 3


def test_strips_chatml_tokens_and_code_fences(tmp_path, reload_node):
    raw = (
        "```json\n"
        + json.dumps({
            "summary": "Wrapped output.",
            "keywords": ["alpha", "beta", "gamma"],
        })
        + "\n```<|im_end|>"
    )
    _install_fake_vllm(raw)
    model_dir = _populate_fake_model_dir(tmp_path)

    from summarizer_vllm import SummarizerVLLM

    node = SummarizerVLLM(model_name=str(model_dir))
    node._name = "test"
    node.transmit = MagicMock()
    node.warmup()
    node.update(_make_window_message("Transcript."))

    out = node.transmit.call_args[0][0].payload
    assert out["summary"] == "Wrapped output."
    assert out["keywords"] == ["alpha", "beta", "gamma"]


def test_missing_model_dir_raises_with_clear_message(tmp_path, reload_node):
    _install_fake_vllm("{}")
    # Path that does not exist
    missing = tmp_path / "models" / "does-not-exist"

    from summarizer_vllm import SummarizerVLLM

    node = SummarizerVLLM(model_name=str(missing))
    node._name = "test"
    with pytest.raises(FileNotFoundError) as exc:
        node.warmup()
    assert "fetch_models.sh" in str(exc.value)


def test_missing_config_json_raises(tmp_path, reload_node):
    _install_fake_vllm("{}")
    # Dir exists but has no config.json (partial download)
    model_dir = tmp_path / "models" / "partial"
    model_dir.mkdir(parents=True)

    from summarizer_vllm import SummarizerVLLM

    node = SummarizerVLLM(model_name=str(model_dir))
    node._name = "test"
    with pytest.raises(FileNotFoundError):
        node.warmup()
