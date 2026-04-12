import json
import re
import time
import typing
import logging
from pathlib import Path

from mlx_lm import load, generate
from juturna.components import Node, Message
from juturna.payloads import ObjectPayload, ControlPayload



class SummarizerMlx(Node[ObjectPayload, ObjectPayload]):
    """Summarizes transcript windows using mlx-lm on Apple Silicon."""

    def __init__(self, model_name: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                 prompt_template_file: str = "summarize_prompt_mlx.txt",
                 num_predict: int = 128,
                 **kwargs):
        super().__init__(**kwargs)
        self._model_name = model_name
        self._mlx_model = None
        self._mlx_tokenizer = None
        self._prompt_template = ""
        self._prompt_file = prompt_template_file
        self._num_predict = num_predict
        self._logger = logging.getLogger(self.__class__.__name__)

    def configure(self): pass

    def warmup(self):
        template_path = Path(__file__).parent / self._prompt_file
        if template_path.exists():
            self._prompt_template = template_path.read_text()
        else:
            self._prompt_template = (
                "Summarize this meeting transcript. "
                "Produce a JSON with 'summary' (2-4 sentences) and 'keywords' (exactly 3).\n\n"
                "Transcript:\n{transcript}"
            )

        self._mlx_model, self._mlx_tokenizer = load(self._model_name)

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
            formatted = self._mlx_tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            content = generate(
                self._mlx_model, self._mlx_tokenizer,
                prompt=formatted, max_tokens=self._num_predict, verbose=False,
            )
            # Strip markdown code fences if present
            content = re.sub(r"^```[a-z]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
            parsed = json.loads(content)
            summary = parsed.get("summary", "")
            keywords = self._ensure_three_keywords(parsed.get("keywords", []))
        except Exception as e:
            self._logger.error(f"LLM call failed: {e}")
            summary = f""
            keywords = []

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
