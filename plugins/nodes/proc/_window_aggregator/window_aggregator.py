import logging
import threading
import time
import typing
from juturna.components import Node, Message
from juturna.payloads import ObjectPayload

logger = logging.getLogger(__name__)


class WindowAggregator(Node[ObjectPayload, ObjectPayload]):
    """Accumulates transcript chunks into fixed-duration windows.

    Includes an inactivity watchdog: if no new chunks arrive for
    ``flush_timeout`` seconds while text is buffered, the partial
    window is flushed automatically.  This handles end-of-stream
    when the upstream source (audio_rtp) cannot signal it.
    """

    def __init__(self, window_duration: float = 300.0,
                 flush_timeout: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self._window_duration = window_duration
        self._flush_timeout = flush_timeout

        self._texts: list[str] = []
        self._window_id: int = 0
        self._window_start: float = 0.0
        self._last_message: Message | None = None
        self._last_chunk_end: float = 0.0
        self._last_chunk_time: float = 0.0

        self._lock = threading.Lock()
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None

    def configure(self): pass

    def warmup(self):
        self._texts = []
        self._window_id = 0
        self._window_start = 0.0
        self._last_message = None
        self._last_chunk_end = 0.0
        self._last_chunk_time = 0.0

    def set_on_config(self, prop: str, value: typing.Any): pass

    def start(self):
        super().start()
        self._last_chunk_time = time.time()
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog, daemon=True)
        self._watchdog_thread.start()

    def stop(self):
        self._watchdog_stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=5)
            self._watchdog_thread = None
        with self._lock:
            self._flush(time.time(), partial=True)
        super().stop()

    def destroy(self): pass

    def _watchdog(self):
        while not self._watchdog_stop.wait(timeout=5.0):
            with self._lock:
                if (self._texts
                        and time.time() - self._last_chunk_time
                        > self._flush_timeout):
                    logger.info(
                        "Inactivity timeout (%.0fs) — flushing partial "
                        "window %d", self._flush_timeout, self._window_id)
                    self._flush(time.time(), partial=True)
                    break

    def _flush(self, wall_time: float, partial: bool = False):
        if not self._texts:
            return
        if partial:
            window_end = self._last_chunk_end
        else:
            window_end = self._window_start + self._window_duration
        payload = ObjectPayload.from_dict({
            "window_id": self._window_id,
            "window_start": self._window_start,
            "window_end": window_end,
            "window_duration": self._window_duration,
            "full_transcript": " ".join(self._texts),
            "trigger_time": wall_time,
        })
        out = Message[ObjectPayload](
            creator=self.name,
            version=self._last_message.version if self._last_message else 1,
            payload=payload,
            timers_from=self._last_message,
        )
        self.transmit(out)
        self._texts = []
        self._window_id += 1
        self._window_start = window_end

    def update(self, message: Message[ObjectPayload]):
        with self._lock:
            self._last_message = message
            novel_text = message.payload.get("novel_text", "")
            chunk_end = message.payload.get("chunk_end", 0.0)
            self._last_chunk_end = chunk_end
            self._last_chunk_time = time.time()

            if novel_text.strip():
                self._texts.append(novel_text)

            if chunk_end >= self._window_start + self._window_duration:
                wall_time = message.payload.get("wall_time", time.time())
                self._flush(wall_time)
