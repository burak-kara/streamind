import time
import typing
from juturna.components import Node, Message
from juturna.payloads import ObjectPayload


class WindowAggregator(Node[ObjectPayload, ObjectPayload]):
    """Accumulates transcript chunks into fixed-duration windows."""

    def __init__(self, window_duration: float = 300.0, **kwargs):
        super().__init__(**kwargs)
        self._window_duration = window_duration
        self._texts: list[str] = []
        self._window_id: int = 0
        self._window_start: float = 0.0

    def configure(self): pass
    def warmup(self):
        self._texts = []
        self._window_id = 0
        self._window_start = 0.0
    def set_on_config(self, prop: str, value: typing.Any): pass
    def start(self): super().start()

    def stop(self):
        self._flush(time.time())
        super().stop()

    def destroy(self): pass

    def _flush(self, wall_time: float):
        if not self._texts:
            return
        window_end = self._window_start + self._window_duration
        payload = ObjectPayload.from_dict({
            "window_id": self._window_id,
            "window_start": self._window_start,
            "window_end": window_end,
            "full_transcript": " ".join(self._texts),
            "trigger_time": wall_time,
        })
        out = Message[ObjectPayload](creator=self.name, version=1, payload=payload)
        self.transmit(out)
        self._texts = []
        self._window_id += 1
        self._window_start = window_end

    def update(self, message: Message[ObjectPayload]):
        novel_text = message.payload.get("novel_text", "")
        chunk_end = message.payload.get("chunk_end", 0.0)

        if novel_text.strip():
            self._texts.append(novel_text)

        if chunk_end >= self._window_start + self._window_duration:
            wall_time = message.payload.get("wall_time", time.time())
            self._flush(wall_time)
