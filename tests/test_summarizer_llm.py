import sys
import os
import json
import time
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'plugins', 'nodes', 'proc', '_summarizer_llm'))

from juturna.components import Message
from juturna.payloads import ObjectPayload


class TestSummarizerLLM:

    def _make_window_message(self, transcript: str, window_id: int = 0) -> Message[ObjectPayload]:
        payload = ObjectPayload.from_dict({
            "window_id": window_id,
            "window_start": window_id * 300.0,
            "window_end": (window_id + 1) * 300.0,
            "full_transcript": transcript,
            "trigger_time": time.time(),
        })
        return Message[ObjectPayload](creator="test", version=1, payload=payload)

    @patch("summarizer_llm.ollama.Client")
    def test_produces_summary_and_keywords(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {
                "content": json.dumps({
                    "summary": "The team discussed project timelines.",
                    "keywords": ["project", "timelines", "discussion"]
                })
            }
        }
        mock_client_cls.return_value = mock_client

        from summarizer_llm import SummarizerLLM

        node = SummarizerLLM(
            endpoint="http://localhost:11434",
            model_name="qwen2.5:7b-instruct",
        )
        node._name = "test_summarizer"
        node.transmit = MagicMock()
        node.warmup()

        msg = self._make_window_message("We discussed project timelines and deadlines.")
        node.update(msg)

        assert node.transmit.call_count == 1
        result = node.transmit.call_args[0][0].payload
        assert "summary" in result
        assert len(result["keywords"]) == 3
        assert isinstance(result["latency"], float)

    @patch("summarizer_llm.ollama.Client")
    def test_malformed_response_pads_keywords(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {
                "content": json.dumps({
                    "summary": "Some summary",
                    "keywords": ["only_one"]
                })
            }
        }
        mock_client_cls.return_value = mock_client

        from summarizer_llm import SummarizerLLM

        node = SummarizerLLM(
            endpoint="http://localhost:11434",
            model_name="qwen2.5:7b-instruct",
        )
        node._name = "test_summarizer"
        node.transmit = MagicMock()
        node.warmup()

        msg = self._make_window_message("Some meeting content.")
        node.update(msg)

        assert node.transmit.call_count == 1
        result = node.transmit.call_args[0][0].payload
        assert len(result["keywords"]) == 3
