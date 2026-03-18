import sys
import os
import numpy as np
import pytest
from unittest.mock import MagicMock

# Add node to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'plugins', 'nodes', 'proc', '_audio_chunker'))

from juturna.components import Message
from juturna.payloads import AudioPayload


class TestAudioChunker:

    def _make_audio_message(self, samples: np.ndarray, start: float, end: float,
                            rate: int = 16000) -> Message[AudioPayload]:
        return Message[AudioPayload](
            creator="test",
            version=1,
            payload=AudioPayload(
                audio=samples,
                sampling_rate=rate,
                channels=1,
                start=start,
                end=end,
            ),
        )

    def test_no_output_before_chunk_filled(self):
        from audio_chunker import AudioChunker

        node = AudioChunker(chunk_duration=5.0, overlap_duration=1.0, sample_rate=16000)
        node._name = "test_chunker"
        node.transmit = MagicMock()
        node.warmup()

        samples = np.zeros(16000, dtype=np.float32)
        msg = self._make_audio_message(samples, start=0.0, end=1.0)
        node.update(msg)

        node.transmit.assert_not_called()

    def test_emits_chunk_after_duration_reached(self):
        from audio_chunker import AudioChunker

        node = AudioChunker(chunk_duration=5.0, overlap_duration=1.0, sample_rate=16000)
        node._name = "test_chunker"
        node.transmit = MagicMock()
        node.warmup()

        for i in range(5):
            samples = np.ones(16000, dtype=np.float32) * (i + 1)
            msg = self._make_audio_message(samples, start=float(i), end=float(i + 1))
            node.update(msg)

        assert node.transmit.call_count == 1
        emitted = node.transmit.call_args[0][0]
        assert emitted.payload.sampling_rate == 16000
        assert len(emitted.payload.audio) == 80000
        assert emitted.payload.start == 0.0
        assert emitted.payload.end == 5.0

    def test_overlap_retained(self):
        from audio_chunker import AudioChunker

        node = AudioChunker(chunk_duration=5.0, overlap_duration=1.0, sample_rate=16000)
        node._name = "test_chunker"
        node.transmit = MagicMock()
        node.warmup()

        for i in range(9):
            samples = np.ones(16000, dtype=np.float32) * (i + 1)
            msg = self._make_audio_message(samples, start=float(i), end=float(i + 1))
            node.update(msg)

        assert node.transmit.call_count == 2
        second_chunk = node.transmit.call_args_list[1][0][0]
        assert second_chunk.payload.start == 4.0
        assert second_chunk.payload.end == 9.0
