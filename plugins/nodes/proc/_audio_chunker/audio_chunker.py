import typing
import numpy as np
from juturna.components import Node, Message
from juturna.payloads import AudioPayload


class AudioChunker(Node[AudioPayload, AudioPayload]):
    """Accumulates audio frames into fixed-duration overlapping chunks."""

    def __init__(self, chunk_duration: float = 5.0, overlap_duration: float = 1.0,
                 sample_rate: int = 16000, **kwargs):
        super().__init__(**kwargs)
        self._chunk_duration = chunk_duration
        self._overlap_duration = overlap_duration
        self._sample_rate = sample_rate
        self._chunk_samples = int(chunk_duration * sample_rate)
        self._overlap_samples = int(overlap_duration * sample_rate)
        self._buffer: np.ndarray = np.array([], dtype=np.float32)
        self._buffer_start: float = 0.0

    def configure(self): pass
    def warmup(self):
        self._buffer = np.array([], dtype=np.float32)
        self._buffer_start = 0.0

    def set_on_config(self, prop: str, value: typing.Any): pass
    def start(self): super().start()
    def stop(self): super().stop()
    def destroy(self): pass

    def update(self, message: Message[AudioPayload]):
        audio = message.payload.audio
        if self._buffer.size == 0:
            self._buffer_start = message.payload.start

        self._buffer = np.concatenate([self._buffer, audio])

        while self._buffer.size >= self._chunk_samples:
            chunk = self._buffer[:self._chunk_samples].copy()
            chunk_end = self._buffer_start + self._chunk_duration

            out = Message[AudioPayload](
                creator=self.name,
                version=message.version,
                payload=AudioPayload(
                    audio=chunk,
                    sampling_rate=self._sample_rate,
                    channels=1,
                    start=self._buffer_start,
                    end=chunk_end,
                ),
                timers_from=message,
            )
            self.transmit(out)

            advance = self._chunk_samples - self._overlap_samples
            self._buffer = self._buffer[advance:]
            self._buffer_start += advance / self._sample_rate
