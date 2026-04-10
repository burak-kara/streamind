import gc
import time
import typing
import logging

import numpy as np
from faster_whisper import WhisperModel

from juturna.components import Node, Message
from juturna.payloads import AudioPayload, ObjectPayload


class TranscriberWhisper(Node[AudioPayload, ObjectPayload]):
    """Transcribes audio chunks using faster-whisper, outputs plain text."""

    def __init__(
        self,
        model_name: str = "small.en",
        language: str = "en",
        device: str = "auto",
        compute_type: str = "int8",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model_name = model_name
        self._language = language
        self._device = device
        self._compute_type = compute_type
        self._model: WhisperModel | None = None

        logging.getLogger("faster_whisper").setLevel(logging.ERROR)

    def configure(self):
        pass

    def warmup(self):
        self._model = WhisperModel(
            self._model_name,
            device=self._device,
            compute_type=self._compute_type,
        )

    def set_on_config(self, prop: str, value: typing.Any):
        pass

    def start(self):
        super().start()

    def stop(self):
        super().stop()

    def destroy(self):
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()

    def update(self, message: Message[AudioPayload]):
        audio = message.payload.audio
        chunk_start = message.payload.start
        chunk_end = message.payload.end

        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            task="transcribe",
            condition_on_previous_text=False,
            vad_filter=True,
        )

        transcript = " ".join(seg.text.strip() for seg in segments)

        payload = ObjectPayload.from_dict({
            "transcript": transcript,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
        })
        out = Message[ObjectPayload](
            creator=self.name,
            version=message.version,
            payload=payload,
            timers_from=message,
        )
        self.transmit(out)
