import sys
import os
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'plugins', 'nodes', 'proc', '_window_aggregator'))

from juturna.components import Message
from juturna.payloads import ObjectPayload


class TestWindowAggregator:

    def _make_novel_message(self, text: str, chunk_end: float) -> Message[ObjectPayload]:
        payload = ObjectPayload.from_dict({
            "novel_text": text,
            "chunk_end": chunk_end,
            "wall_time": 1000000.0 + chunk_end,
        })
        return Message[ObjectPayload](creator="test", version=1, payload=payload)

    def test_no_output_before_window_full(self):
        from window_aggregator import WindowAggregator

        node = WindowAggregator(window_duration=300.0)
        node._name = "test_agg"
        node.transmit = MagicMock()
        node.warmup()

        node.update(self._make_novel_message("hello world", chunk_end=5.0))
        node.transmit.assert_not_called()

    def test_emits_at_window_boundary(self):
        from window_aggregator import WindowAggregator

        node = WindowAggregator(window_duration=10.0)
        node._name = "test_agg"
        node.transmit = MagicMock()
        node.warmup()

        node.update(self._make_novel_message("hello", chunk_end=5.0))
        node.update(self._make_novel_message("world", chunk_end=10.0))

        assert node.transmit.call_count == 1
        result = node.transmit.call_args[0][0].payload
        assert result["full_transcript"] == "hello world"
        assert result["window_id"] == 0
        assert result["window_start"] == 0.0
        assert result["window_end"] == 10.0

    def test_second_window(self):
        from window_aggregator import WindowAggregator

        node = WindowAggregator(window_duration=10.0)
        node._name = "test_agg"
        node.transmit = MagicMock()
        node.warmup()

        node.update(self._make_novel_message("first", chunk_end=5.0))
        node.update(self._make_novel_message("window", chunk_end=10.0))
        node.update(self._make_novel_message("second", chunk_end=15.0))
        node.update(self._make_novel_message("window", chunk_end=20.0))

        assert node.transmit.call_count == 2
        second = node.transmit.call_args_list[1][0][0].payload
        assert second["window_id"] == 1
        assert second["full_transcript"] == "second window"

    def test_empty_text_not_accumulated(self):
        from window_aggregator import WindowAggregator

        node = WindowAggregator(window_duration=10.0)
        node._name = "test_agg"
        node.transmit = MagicMock()
        node.warmup()

        node.update(self._make_novel_message("hello", chunk_end=5.0))
        node.update(self._make_novel_message("", chunk_end=7.0))
        node.update(self._make_novel_message("world", chunk_end=10.0))

        result = node.transmit.call_args[0][0].payload
        assert result["full_transcript"] == "hello world"
