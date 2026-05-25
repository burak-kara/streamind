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
        self._audio_buf: np.ndarray = np.array([], dtype=np.float32)
        self._audio_buf_start: float = 0.0

    def configure(self): pass
    def warmup(self):
        self._audio_buf = np.array([], dtype=np.float32)
        self._audio_buf_start = 0.0

    def set_on_config(self, prop: str, value: typing.Any): pass
    def start(self): super().start()

    def stop(self):
        if self._audio_buf.size > 0:
            chunk = self._audio_buf.copy()
            chunk_end = self._audio_buf_start + len(chunk) / self._sample_rate
            out = Message[AudioPayload](
                creator=self.name,
                version=0,
                payload=AudioPayload(
                    audio=chunk,
                    sampling_rate=self._sample_rate,
                    channels=1,
                    start=self._audio_buf_start,
                    end=chunk_end,
                ),
            )
            self.transmit(out)
            self._audio_buf = np.array([], dtype=np.float32)
        super().stop()

    def destroy(self): pass

    def update(self, message: Message[AudioPayload]):
        audio = message.payload.audio
        if self._audio_buf.size == 0:
            self._audio_buf_start = message.payload.start

        self._audio_buf = np.concatenate([self._audio_buf, audio])

        while self._audio_buf.size >= self._chunk_samples:
            chunk = self._audio_buf[:self._chunk_samples].copy()
            chunk_end = self._audio_buf_start + self._chunk_duration

            out = Message[AudioPayload](
                creator=self.name,
                version=message.version,
                payload=AudioPayload(
                    audio=chunk,
                    sampling_rate=self._sample_rate,
                    channels=1,
                    start=self._audio_buf_start,
                    end=chunk_end,
                ),
                timers_from=message,
            )
            self.transmit(out)

            advance = self._chunk_samples - self._overlap_samples
            self._audio_buf = self._audio_buf[advance:]
            self._audio_buf_start += advance / self._sample_rate
