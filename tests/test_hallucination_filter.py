import sys
import os
import tomllib
from unittest.mock import MagicMock

_NODE_DIR = os.path.join(os.path.dirname(__file__), '..', 'plugins', 'nodes', 'proc', '_hallucination_filter')
sys.path.insert(0, _NODE_DIR)

with open(os.path.join(_NODE_DIR, 'config.toml'), 'rb') as _f:
    _DEFAULTS = tomllib.load(_f)['arguments']

from juturna.components import Message
from juturna.payloads import ObjectPayload


def _make_message(transcript: str, chunk_start: float = 0.0, chunk_end: float = 5.0) -> Message[ObjectPayload]:
    payload = ObjectPayload.from_dict({
        "transcript": transcript,
        "chunk_start": chunk_start,
        "chunk_end": chunk_end,
    })
    return Message[ObjectPayload](creator="test", version=1, payload=payload)


class TestHallucinationFilter:

    def _node(self, extra_patterns=None):
        from hallucination_filter import HallucinationFilter
        node = HallucinationFilter(
            exact_hallucinations=_DEFAULTS['exact_hallucinations'],
            extra_patterns=extra_patterns or [],
        )
        node._name = "test_filter"
        node.transmit = MagicMock()
        return node

    def test_clean_transcript_passes_through(self):
        node = self._node()
        msg = _make_message("The meeting will start in five minutes.")
        node.update(msg)
        assert node.transmit.call_count == 1
        assert node.transmit.call_args[0][0] is msg  # same object, no copy

    def test_known_hallucination_dropped(self):
        node = self._node()
        for phrase in ["Thank you.", "Thanks for watching.", "you", "[BLANK_AUDIO]", "[silence]"]:
            node.transmit.reset_mock()
            node.update(_make_message(phrase))
            assert node.transmit.call_count == 0, f"Expected drop for: {phrase!r}"

    def test_case_insensitive_hallucination(self):
        node = self._node()
        node.update(_make_message("THANK YOU."))
        assert node.transmit.call_count == 0

    def test_empty_transcript_dropped(self):
        node = self._node()
        node.update(_make_message(""))
        assert node.transmit.call_count == 0

    def test_whitespace_only_dropped(self):
        node = self._node()
        node.update(_make_message("   "))
        assert node.transmit.call_count == 0

    def test_bracketed_tag_stripped_and_passed(self):
        node = self._node()
        node.update(_make_message("Real speech here. [Music] More speech."))
        assert node.transmit.call_count == 1
        result = node.transmit.call_args[0][0].payload
        assert "[Music]" not in result["transcript"]
        assert "Real speech here." in result["transcript"]

    def test_parenthetical_tag_stripped_and_passed(self):
        node = self._node()
        node.update(_make_message("(Applause) Let's get started."))
        assert node.transmit.call_count == 1
        result = node.transmit.call_args[0][0].payload
        assert "(Applause)" not in result["transcript"]
        assert "Let's get started." in result["transcript"]

    def test_pure_bracketed_tag_dropped(self):
        node = self._node()
        node.update(_make_message("[Music]"))
        assert node.transmit.call_count == 0

    def test_pure_parenthetical_dropped(self):
        node = self._node()
        node.update(_make_message("(laughter)"))
        assert node.transmit.call_count == 0

    def test_chunk_timestamps_preserved_after_strip(self):
        node = self._node()
        node.update(_make_message("[Music] Something real was said.", chunk_start=10.0, chunk_end=15.0))
        assert node.transmit.call_count == 1
        result = node.transmit.call_args[0][0].payload
        assert result["chunk_start"] == 10.0
        assert result["chunk_end"] == 15.0

    def test_extra_patterns_dropped(self):
        node = self._node(extra_patterns=["custom noise phrase"])
        node.update(_make_message("custom noise phrase"))
        assert node.transmit.call_count == 0

    def test_extra_patterns_case_insensitive(self):
        node = self._node(extra_patterns=["custom noise phrase"])
        node.update(_make_message("CUSTOM NOISE PHRASE"))
        assert node.transmit.call_count == 0

    def test_real_transcript_with_trailing_hallucination(self):
        # The hallucination filter strips inline tags but does NOT remove
        # trailing known-exact phrases that appear as part of real sentences.
        # This is intentional — we only strip bracket/paren tags, not sentences.
        node = self._node()
        msg = _make_message("We discussed the roadmap. Thank you for joining.")
        node.update(msg)
        assert node.transmit.call_count == 1
        result = node.transmit.call_args[0][0].payload
        assert "roadmap" in result["transcript"]
