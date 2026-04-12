import sys
import os
import json
import tempfile
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'plugins', 'nodes', 'sink', '_result_transmitter'))

from juturna.components import Message
from juturna.payloads import ObjectPayload


class TestResultTransmitter:

    def _make_result_message(self, window_id: int = 0,
                              model_name: str = "test-model") -> Message[ObjectPayload]:
        payload = ObjectPayload.from_dict({
            "window_id": window_id,
            "window_start": 0.0,
            "window_end": 300.0,
            "summary": "The team discussed project deadlines.",
            "keywords": ["project", "deadlines", "team"],
            "latency": 1.5,
            "model_name": model_name,
        })
        return Message[ObjectPayload](creator="test", version=1, payload=payload)

    def test_saves_result_to_file(self):
        from result_transmitter import ResultTransmitter

        with tempfile.TemporaryDirectory() as tmpdir:
            node = ResultTransmitter(
                destination_endpoint="http://example.com/submit",
                results_dir=tmpdir,
            )
            node._name = "test_tx"
            node.warmup()

            msg = self._make_result_message(window_id=0)
            node.update(msg)

            result_file = os.path.join(tmpdir, "test-model", "300", "window_0.json")
            assert os.path.exists(result_file)
            with open(result_file) as f:
                data = json.load(f)
            # Verify challenge-compliant output format
            assert data["from"] == 0.0
            assert data["to"] == 300.0
            assert data["summary"] == "The team discussed project deadlines."
            assert data["keywords"] == ["project", "deadlines", "team"]
            assert len(data["keywords"]) == 3
            assert data["proc_time"] == 1.5
            # Verify old keys are not present
            assert "window_start" not in data
            assert "window_end" not in data
            assert "latency" not in data

    @patch("result_transmitter.httpx.post")
    def test_posts_to_endpoint(self, mock_post):
        from result_transmitter import ResultTransmitter

        mock_post.return_value = MagicMock(status_code=200)

        with tempfile.TemporaryDirectory() as tmpdir:
            node = ResultTransmitter(
                destination_endpoint="http://example.com/submit",
                results_dir=tmpdir,
            )
            node._name = "test_tx"
            node.warmup()

            msg = self._make_result_message()
            node.update(msg)

            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            assert "http://example.com/submit" in str(call_kwargs)
