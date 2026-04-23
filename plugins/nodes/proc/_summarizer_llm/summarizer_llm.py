import importlib.util
import json
import re
import time
import typing
import logging
from pathlib import Path

import ollama
from juturna.components import Node, Message
from juturna.payloads import ObjectPayload, ControlPayload


# Load the shared keyword-handling module by path. Juturna loads this file by
# path too (not as part of a package), so a regular import cannot reach the
# sibling `_summarizer_common` directory.
_kw_spec = importlib.util.spec_from_file_location(
    "summarizer_keywords",
    Path(__file__).resolve().parent.parent / "_summarizer_common" / "keywords.py",
)
_kw_module = importlib.util.module_from_spec(_kw_spec)
assert _kw_spec.loader is not None
_kw_spec.loader.exec_module(_kw_module)
BANNED_KEYWORDS = _kw_module.BANNED_KEYWORDS
_ensure_three_keywords_impl = _kw_module.ensure_three_keywords



class SummarizerLLM(Node[ObjectPayload, ObjectPayload]):
    """Summarizes transcript windows using a local LLM via Ollama."""

    def __init__(self, endpoint: str = "http://127.0.0.1:11434",
                 model_name: str = "qwen3.5:9b-16k",
                 prompt_template_file: str = "summarize_prompt_ollama.txt",
                 num_ctx: int = 2048,
                 num_predict: int = 128,
                 **kwargs):
        super().__init__(**kwargs)
        self._endpoint = endpoint
        self._model_name = model_name
        self._client: ollama.Client | None = None
        self._prompt_template = ""
        self._prompt_file = prompt_template_file
        self._num_ctx = num_ctx
        self._num_predict = num_predict
        self._logger = logging.getLogger(self.__class__.__name__)

    def configure(self): pass

    def warmup(self):
        template_path = Path(__file__).parent / self._prompt_file
        if template_path.exists():
            self._prompt_template = template_path.read_text()
        else:
            self._logger.error(f"Prompt template file not found: {template_path}. Using default template.")

        self._client = ollama.Client(host=self._endpoint)
        try:
            self._client.chat(
                model=self._model_name,
                messages=[{"role": "user", "content": "Say hello."}],
                options={"num_predict": 5},
                think=False,
            )
        except Exception as e:
            self._logger.error(f"Model warmup failed: {e}")

    def set_on_config(self, prop: str, value: typing.Any): pass
    def start(self): super().start()

    def stop(self):
        # Drain any pending messages before stopping (handles flush-during-stop race)
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

    def _ensure_three_keywords(self, keywords: list, transcript: str = "") -> list[str]:
        return _ensure_three_keywords_impl(keywords, transcript)

    def update(self, message: Message[ObjectPayload]):
        transcript = message.payload.get("full_transcript", "")
        trigger_time = message.payload.get("trigger_time", time.time())

        prompt = self._prompt_template.format(transcript=transcript)

        try:
            response = self._client.chat(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": self._num_predict, "num_ctx": self._num_ctx},
                think=False,
            )
            content = response.message.content.strip()
            # Strip Qwen/ChatML special tokens like <|im_end|>, <|endoftext|>.
            content = re.sub(r"<\|[^|]+\|>", "", content)
            # Strip markdown code fences if present
            content = re.sub(r"^```[a-z]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
            content = content.strip()
            # Extract the outermost JSON object — tolerates trailing stop tokens
            # or commentary the model may emit after the JSON.
            obj_match = re.search(r"\{.*\}", content, re.DOTALL)
            if obj_match is None:
                raise ValueError(f"No JSON object found in LLM output: {content[:120]!r}")
            parsed = json.loads(obj_match.group(0))
            summary = parsed.get("summary", "")
            keywords = self._ensure_three_keywords(parsed.get("keywords", []), transcript)
        except Exception as e:
            self._logger.error(f"LLM call failed: {e}")
            summary = ""
            keywords = self._ensure_three_keywords([], transcript)

        latency = time.time() - trigger_time

        payload = ObjectPayload.from_dict({
            "window_id": message.payload.get("window_id", 0),
            "window_start": message.payload.get("window_start", 0.0),
            "window_end": message.payload.get("window_end", 0.0),
            "summary": summary,
            "keywords": keywords,
            "latency": latency,
            "model_name": self._model_name,
            "full_transcript": transcript,
        })
        out = Message[ObjectPayload](
            creator=self.name,
            version=message.version,
            payload=payload,
            timers_from=message,
        )
        self.transmit(out)
