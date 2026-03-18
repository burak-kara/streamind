import json
import time
import typing
import logging
from pathlib import Path

import ollama
from juturna.components import Node, Message
from juturna.payloads import ObjectPayload


_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
        },
    },
    "required": ["summary", "keywords"],
}


class SummarizerLLM(Node[ObjectPayload, ObjectPayload]):
    """Summarizes transcript windows using a local LLM via Ollama."""

    def __init__(self, endpoint: str = "http://127.0.0.1:11434",
                 model_name: str = "qwen2.5:7b-instruct",
                 prompt_template_file: str = "summarize_prompt.txt",
                 **kwargs):
        super().__init__(**kwargs)
        self._endpoint = endpoint
        self._model_name = model_name
        self._client: ollama.Client | None = None
        self._prompt_template = ""
        self._prompt_file = prompt_template_file
        self._logger = logging.getLogger(self.__class__.__name__)

    def configure(self): pass

    def warmup(self):
        self._client = ollama.Client(host=self._endpoint)
        template_path = Path(__file__).parent / self._prompt_file
        if template_path.exists():
            self._prompt_template = template_path.read_text()
        else:
            self._prompt_template = (
                "Summarize this meeting transcript. "
                "Produce a JSON with 'summary' (2-4 sentences) and 'keywords' (exactly 3).\n\n"
                "Transcript:\n{transcript}"
            )
        try:
            self._client.chat(
                model=self._model_name,
                messages=[{"role": "user", "content": "Say hello."}],
                options={"num_predict": 1},
            )
        except Exception as e:
            self._logger.warning(f"Model warmup failed: {e}")

    def set_on_config(self, prop: str, value: typing.Any): pass
    def start(self): super().start()
    def stop(self): super().stop()
    def destroy(self): pass

    def _ensure_three_keywords(self, keywords: list) -> list[str]:
        kw = [str(k) for k in keywords]
        while len(kw) < 3:
            kw.append("general")
        return kw[:3]

    def update(self, message: Message[ObjectPayload]):
        transcript = message.payload.get("full_transcript", "")
        trigger_time = message.payload.get("trigger_time", time.time())

        prompt = self._prompt_template.format(transcript=transcript)

        try:
            response = self._client.chat(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                format=_OUTPUT_SCHEMA,
            )
            content = response["message"]["content"]
            parsed = json.loads(content)
            summary = parsed.get("summary", "")
            keywords = self._ensure_three_keywords(parsed.get("keywords", []))
        except Exception as e:
            self._logger.error(f"LLM call failed: {e}")
            summary = f"Transcript segment ({len(transcript.split())} words)"
            keywords = ["meeting", "discussion", "general"]

        latency = time.time() - trigger_time

        payload = ObjectPayload.from_dict({
            "window_id": message.payload.get("window_id", 0),
            "window_start": message.payload.get("window_start", 0.0),
            "window_end": message.payload.get("window_end", 0.0),
            "summary": summary,
            "keywords": keywords,
            "latency": latency,
        })
        out = Message[ObjectPayload](
            creator=self.name,
            version=message.version,
            payload=payload,
            timers_from=message,
        )
        self.transmit(out)
