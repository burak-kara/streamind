import json
import re
import time
import typing
import logging
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from juturna.components import Node, Message
from juturna.payloads import ObjectPayload, ControlPayload



class SummarizerMlx(Node[ObjectPayload, ObjectPayload]):
    """Summarizes transcript windows using mlx-lm on Apple Silicon."""

    def __init__(self, model_name: str = "",
                 prompt_template_file: str = "",
                 num_predict: int = 128,
                 temp: float = 0.0,
                 top_p: float = 1.0,
                 repetition_penalty: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self._model_name = model_name
        self._mlx_model = None
        self._mlx_tokenizer = None
        self._prompt_template = ""
        self._prompt_file = prompt_template_file
        self._num_predict = num_predict
        self._temp = temp
        self._top_p = top_p
        self._repetition_penalty = repetition_penalty
        self._logger = logging.getLogger(self.__class__.__name__)

    def configure(self): pass

    def warmup(self):
        template_path = Path(__file__).parent / self._prompt_file
        if template_path.exists():
            self._prompt_template = template_path.read_text()
        else:
            self._logger.error(f"Prompt template file not found: {template_path}")
            raise FileNotFoundError(f"Prompt template file not found: {template_path}")

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

        raw_llm_output = ""
        try:
            template_kwargs = dict(tokenize=False, add_generation_prompt=True)
            try:
                formatted = self._mlx_tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    enable_thinking=False,
                    **template_kwargs,
                )
            except TypeError:
                # Tokenizer doesn't support enable_thinking; fall back to plain template
                self._logger.warning("Tokenizer does not support enable_thinking; using fallback template formatting")
                formatted = self._mlx_tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    **template_kwargs,
                )
            sampler = make_sampler(temp=self._temp, top_p=self._top_p)
            logits_processors = make_logits_processors(
                repetition_penalty=self._repetition_penalty
            )
            content = generate(
                self._mlx_model, self._mlx_tokenizer,
                prompt=formatted,
                max_tokens=self._num_predict,
                sampler=sampler,
                logits_processors=logits_processors,
                verbose=False,
            )
            raw_llm_output = content
            self._logger.debug(f"Raw LLM output: {repr(content)}")
            # Strip thinking blocks (Qwen3 may emit <think>...</think> despite /no_think)
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            # Strip markdown code fences if present
            content = re.sub(r"^```[a-z]*\n?", "", content, flags=re.MULTILINE)
            content = re.sub(r"\n?```$", "", content, flags=re.MULTILINE)
            content = content.strip()
            if not content:
                raise ValueError("LLM returned empty content")
            parsed = json.loads(content)
            summary = parsed.get("summary", "")
            keywords = self._ensure_three_keywords(parsed.get("keywords", []))
        except Exception as e:
            self._logger.error(f"LLM call failed: {e}")
            summary = ""
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
            "raw_llm_output": raw_llm_output,
        })
        out = Message[ObjectPayload](
            creator=self.name,
            version=message.version,
            payload=payload,
            timers_from=message,
        )
        self.transmit(out)
