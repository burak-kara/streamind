import sys
import os
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'plugins', 'nodes', 'proc', '_novel_extractor'))

from juturna.components import Message
from juturna.payloads import ObjectPayload


class TestNovelExtractor:

    def _make_transcript_message(self, transcript: str, chunk_start: float,
                                  chunk_end: float) -> Message[ObjectPayload]:
        payload = ObjectPayload.from_dict({
            "transcript": transcript,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
        })
        return Message[ObjectPayload](creator="test", version=1, payload=payload)

    def test_first_chunk_passes_through(self):
        from novel_extractor import NovelExtractor

        node = NovelExtractor(max_overlap_words=10, similarity_threshold=0.8)
        node._name = "test_extractor"
        node.transmit = MagicMock()
        node.warmup()

        msg = self._make_transcript_message("hello world this is a test", 0.0, 5.0)
        node.update(msg)

        assert node.transmit.call_count == 1
        result = node.transmit.call_args[0][0].payload
        assert result["novel_text"] == "hello world this is a test"

    def test_exact_overlap_removed(self):
        from novel_extractor import NovelExtractor

        node = NovelExtractor(max_overlap_words=10, similarity_threshold=0.8)
        node._name = "test_extractor"
        node.transmit = MagicMock()
        node.warmup()

        msg1 = self._make_transcript_message("the quick brown fox jumps", 0.0, 5.0)
        node.update(msg1)

        msg2 = self._make_transcript_message("fox jumps over the lazy dog", 4.0, 9.0)
        node.update(msg2)

        result = node.transmit.call_args[0][0].payload
        assert result["novel_text"] == "over the lazy dog"

    def test_no_overlap(self):
        from novel_extractor import NovelExtractor

        node = NovelExtractor(max_overlap_words=10, similarity_threshold=0.8)
        node._name = "test_extractor"
        node.transmit = MagicMock()
        node.warmup()

        msg1 = self._make_transcript_message("hello world", 0.0, 5.0)
        node.update(msg1)

        msg2 = self._make_transcript_message("completely different text", 5.0, 10.0)
        node.update(msg2)

        result = node.transmit.call_args[0][0].payload
        assert result["novel_text"] == "completely different text"

    def test_empty_transcript(self):
        from novel_extractor import NovelExtractor

        node = NovelExtractor(max_overlap_words=10, similarity_threshold=0.8)
        node._name = "test_extractor"
        node.transmit = MagicMock()
        node.warmup()

        msg = self._make_transcript_message("", 0.0, 5.0)
        node.update(msg)

        result = node.transmit.call_args[0][0].payload
        assert result["novel_text"] == ""

    def test_fuzzy_overlap(self):
        from novel_extractor import NovelExtractor

        node = NovelExtractor(max_overlap_words=10, similarity_threshold=0.6)
        node._name = "test_extractor"
        node.transmit = MagicMock()
        node.warmup()

        msg1 = self._make_transcript_message("going to the store today", 0.0, 5.0)
        node.update(msg1)

        msg2 = self._make_transcript_message("gonna the store today and buy food", 4.0, 9.0)
        node.update(msg2)

        result = node.transmit.call_args[0][0].payload
        assert "buy food" in result["novel_text"]
