import sys
import os
import time
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

    def test_stop_flushes_partial_window(self):
        from window_aggregator import WindowAggregator

        node = WindowAggregator(window_duration=300.0)
        node._name = "test_agg"
        node.transmit = MagicMock()
        node.warmup()

        node.update(self._make_novel_message("partial content", chunk_end=83.6))
        assert node.transmit.call_count == 0

        node.stop()

        assert node.transmit.call_count == 1
        result = node.transmit.call_args[0][0].payload
        assert result["full_transcript"] == "partial content"
        assert result["window_id"] == 0
        assert result["window_end"] == 83.6

    def test_partial_window_end_uses_last_chunk_end(self):
        from window_aggregator import WindowAggregator

        node = WindowAggregator(window_duration=300.0)
        node._name = "test_agg"
        node.transmit = MagicMock()
        node.warmup()

        node.update(self._make_novel_message("first", chunk_end=5.0))
        node.update(self._make_novel_message("window", chunk_end=300.0))
        node.update(self._make_novel_message("leftover", chunk_end=310.5))

        assert node.transmit.call_count == 1
        assert node.transmit.call_args[0][0].payload["window_end"] == 300.0

        node.stop()

        assert node.transmit.call_count == 2
        partial = node.transmit.call_args_list[1][0][0].payload
        assert partial["window_end"] == 310.5
        assert partial["window_start"] == 300.0

    def test_watchdog_flushes_after_timeout(self):
        from window_aggregator import WindowAggregator

        node = WindowAggregator(window_duration=300.0, flush_timeout=0.5)
        node._name = "test_agg"
        node.transmit = MagicMock()
        node.warmup()
        node.start()

        try:
            node.update(self._make_novel_message("timed out content", chunk_end=42.0))
            assert node.transmit.call_count == 0

            time.sleep(6.0)

            assert node.transmit.call_count == 1
            result = node.transmit.call_args[0][0].payload
            assert result["full_transcript"] == "timed out content"
            assert result["window_end"] == 42.0
        finally:
            node._watchdog_stop.set()
            node._watchdog_thread.join(timeout=2)

    def test_stop_no_double_flush(self):
        from window_aggregator import WindowAggregator

        node = WindowAggregator(window_duration=300.0)
        node._name = "test_agg"
        node.transmit = MagicMock()
        node.warmup()

        node.stop()
        assert node.transmit.call_count == 0
