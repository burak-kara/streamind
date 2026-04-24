import importlib.util
import json
import logging
import queue
import sys
import threading
import typing
from pathlib import Path

import ollama
from juturna.components import Node, Message
from juturna.payloads import ObjectPayload, ControlPayload


# Load the shared scorer by file path. Juturna loads node files by path (not as
# part of a package), so the sibling `_judge_common` dir is unreachable via a
# regular import. Mirrors the pattern used by the summarizer nodes.
# The sys.modules registration is required so @dataclass inside the module
# can resolve its own __module__ in Python 3.12+.
_scorer_spec = importlib.util.spec_from_file_location(
    "judge_scorer",
    Path(__file__).resolve().parent.parent / "_judge_common" / "scorer.py",
)
_scorer = importlib.util.module_from_spec(_scorer_spec)
assert _scorer_spec.loader is not None
sys.modules["judge_scorer"] = _scorer
_scorer_spec.loader.exec_module(_scorer)


_SHUTDOWN = object()


class JudgeLLM(Node[ObjectPayload, ObjectPayload]):
    """Async LLM-as-judge sink that scores summarizer output offline.

    Forks from the summarizer in parallel with `result_transmitter`. The live
    transmitter path is never blocked: incoming messages are dropped into a
    bounded queue and a background worker thread runs the judge. If the judge
    falls behind, the oldest unscored window is dropped and a warning logged.
    """

    def __init__(self,
                 endpoint: str = "http://127.0.0.1:11434",
                 model_name: str = "qwen3.5:4b",
                 results_dir: str = "./results",
                 num_ctx: int = 4096,
                 num_predict: int = 256,
                 queue_size: int = 8,
                 worker_join_timeout: float = 10.0,
                 **kwargs):
        super().__init__(**kwargs)
        self._endpoint = endpoint
        self._model_name = model_name
        self._results_dir = Path(results_dir)
        self._num_ctx = num_ctx
        self._num_predict = num_predict
        self._queue_size = max(1, int(queue_size))
        self._worker_join_timeout = worker_join_timeout

        self._client: ollama.Client | None = None
        self._work_queue: queue.Queue = queue.Queue(maxsize=self._queue_size)
        self._worker: threading.Thread | None = None
        self._logger = logging.getLogger(self.__class__.__name__)

    def configure(self): pass

    def warmup(self):
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._client = ollama.Client(host=self._endpoint)
        # Don't probe the judge model at warmup — failing here would take down
        # the whole pipeline. A judge outage degrades evaluation, not the live
        # submission path.
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=f"judge-worker-{self._name}",
            daemon=True,
        )
        self._worker.start()

    def set_on_config(self, prop: str, value: typing.Any): pass
    def start(self): super().start()

    def stop(self):
        # Drain any pending Juturna messages first so we don't lose windows
        # the framework hasn't delivered to update() yet.
        import queue as _q
        while True:
            try:
                msg = self._queue.get_nowait()
            except _q.Empty:
                break
            if msg is not None and not isinstance(msg.payload, ControlPayload):
                self.update(msg)

        # Drain the judge work queue so the shutdown sentinel always has room,
        # and so we don't waste the join timeout scoring windows we're about
        # to abandon anyway.
        while True:
            try:
                self._work_queue.get_nowait()
            except _q.Empty:
                break

        if self._worker is not None:
            try:
                self._work_queue.put_nowait(_SHUTDOWN)
            except _q.Full:
                # Worker must have re-filled the queue from something we
                # couldn't drain — fall back to a bounded blocking put.
                try:
                    self._work_queue.put(_SHUTDOWN, timeout=self._worker_join_timeout)
                except _q.Full:
                    self._logger.warning("Could not enqueue judge shutdown sentinel")
            self._worker.join(timeout=self._worker_join_timeout)
            if self._worker.is_alive():
                self._logger.warning(
                    "Judge worker did not finish within %.1fs; abandoning",
                    self._worker_join_timeout,
                )
        super().stop()

    def destroy(self): pass

    @staticmethod
    def _sanitize_model_name(name: str) -> str:
        return name.replace(":", "-").replace("/", "_")

    def update(self, message: Message[ObjectPayload]):
        # Pull only the fields we need so we don't pin large objects in queue.
        item = {
            "window_id": message.payload.get("window_id", 0),
            "window_start": message.payload.get("window_start", 0.0),
            "window_end": message.payload.get("window_end", 0.0),
            "summary": message.payload.get("summary", ""),
            "keywords": list(message.payload.get("keywords", [])),
            "transcript": message.payload.get("full_transcript", ""),
            "proc_time": float(message.payload.get("latency", 0.0)),
            "summarizer_model": message.payload.get("model_name", "unknown"),
        }

        try:
            self._work_queue.put_nowait(item)
        except queue.Full:
            # Drop-oldest: keep the newest window so live monitoring stays
            # current even when the judge is overloaded.
            try:
                dropped = self._work_queue.get_nowait()
                if isinstance(dropped, dict):
                    self._logger.warning(
                        "Judge queue full; dropping oldest window %s",
                        dropped.get("window_id"),
                    )
            except queue.Empty:
                pass
            try:
                self._work_queue.put_nowait(item)
            except queue.Full:
                self._logger.warning(
                    "Judge queue still full after drop; skipping window %s",
                    item["window_id"],
                )

    def _worker_loop(self):
        while True:
            item = self._work_queue.get()
            if item is _SHUTDOWN:
                return
            try:
                self._score_and_write(item)
            except Exception as e:
                self._logger.error(
                    "Judge scoring failed for window %s: %s",
                    item.get("window_id"), e,
                )

    def _score_and_write(self, item: dict):
        prompt = _scorer.build_prompt(
            transcript=item["transcript"],
            summary=item["summary"],
            keywords=item["keywords"],
        )
        response = self._client.chat(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_predict": self._num_predict,
                "num_ctx": self._num_ctx,
                "temperature": 0.0,
            },
            think=False,
        )
        content = response.message.content
        b_breakdown, keyword_flags = _scorer.parse_judge_response(content)
        score = _scorer.compute_score(b_breakdown, keyword_flags, item["proc_time"])

        window_duration = int(round(item["window_end"] - item["window_start"]))
        safe_model = self._sanitize_model_name(item["summarizer_model"])
        out_dir = self._results_dir / safe_model / str(window_duration) / "judge"
        out_dir.mkdir(parents=True, exist_ok=True)

        record = {
            "window_id": item["window_id"],
            "from": item["window_start"],
            "to": item["window_end"],
            "summary": item["summary"],
            "keywords": item["keywords"],
            "proc_time": item["proc_time"],
            "judge_model": self._model_name,
            **score.to_dict(),
        }
        out_path = out_dir / f"window_{item['window_id']}.json"
        out_path.write_text(json.dumps(record, indent=2))
        self._logger.info(
            "Judged window %s: B=%d K=%d L=%.2f C=%.2f -> %s",
            item["window_id"], score.b, score.k, score.l, score.c, out_path,
        )
