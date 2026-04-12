import json
import typing
import logging
from pathlib import Path

import httpx
from juturna.components import Node, Message
from juturna.payloads import ObjectPayload, ControlPayload


class ResultTransmitter(Node[ObjectPayload, ObjectPayload]):
    """Saves results to filesystem and POSTs to challenge endpoint."""

    def __init__(self, destination_endpoint: str = "", results_dir: str = "./results",
                 timeout: int = 10, **kwargs):
        super().__init__(**kwargs)
        self._endpoint = destination_endpoint
        self._results_dir = Path(results_dir)
        self._timeout = timeout
        self._logger = logging.getLogger(self.__class__.__name__)

    def configure(self): pass

    def warmup(self):
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def set_on_config(self, prop: str, value: typing.Any): pass
    def start(self): super().start()

    def stop(self):
        import queue as _q
        while True:
            try:
                msg = self._queue.get_nowait()
            except _q.Empty:
                break
            if msg is not None and not isinstance(msg.payload, ControlPayload):
                self.update(msg)
        super().stop()
    def destroy(self): pass

    def update(self, message: Message[ObjectPayload]):
        window_id = message.payload.get("window_id", 0)
        result = {
            "from": message.payload.get("window_start", 0.0),
            "to": message.payload.get("window_end", 0.0),
            "summary": message.payload.get("summary", ""),
            "keywords": message.payload.get("keywords", []),
            "proc_time": message.payload.get("latency", 0.0),
        }

        filepath = self._results_dir / f"window_{window_id}.json"
        filepath.write_text(json.dumps(result, indent=2))
        self._logger.info(f"Saved result to {filepath}")

        if self._endpoint:
            try:
                resp = httpx.post(
                    self._endpoint,
                    json=result,
                    timeout=self._timeout,
                )
                self._logger.info(f"POST {self._endpoint} -> {resp.status_code}")
            except Exception as e:
                self._logger.error(f"POST failed: {e}")
