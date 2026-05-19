import importlib.util
import json
import logging
import queue as _q
import re
import time
import typing
from pathlib import Path

from juturna.components import Node, Message
from juturna.payloads import ObjectPayload, ControlPayload


_kw_spec = importlib.util.spec_from_file_location(
    "summarizer_keywords",
    Path(__file__).resolve().parent.parent / "_summarizer_common" / "keywords.py",
)
_kw_module = importlib.util.module_from_spec(_kw_spec)
assert _kw_spec.loader is not None
_kw_spec.loader.exec_module(_kw_module)
BANNED_KEYWORDS = _kw_module.BANNED_KEYWORDS
_ensure_three_keywords_impl = _kw_module.ensure_three_keywords


class SummarizerVLLM(Node[ObjectPayload, ObjectPayload]):
    """Summarizes transcript windows via vLLM in-process inference.

    The summarizer loads model weights from a **local filesystem path** —
    never an HF id at runtime. Weights ship with the submission image,
    populated either by `tools/fetch_models.sh` (dev) or by the Dockerfile
    at build time. Refuses to start if the model directory is missing so
    pipeline failure mode is loud and obvious.
    """

    def __init__(self,
                 model_name: str,
                 prompt_template_file: str = "summarize_prompt.txt",
                 dtype: str = "float16",
                 gpu_memory_utilization: float = 0.85,
                 max_model_len: int = 2048,
                 max_tokens: int = 150,
                 temperature: float = 0.3,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.05,
                 enforce_eager: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self._model_name = model_name
        self._prompt_file = prompt_template_file
        self._dtype = dtype
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._repetition_penalty = repetition_penalty
        self._enforce_eager = enforce_eager

        self._llm = None
        self._sampling_params = None
        self._prompt_template = ""
        self._logger = logging.getLogger(self.__class__.__name__)

    def configure(self): pass

    def warmup(self):
        model_path = Path(self._model_name)
        if not model_path.exists() or not (model_path / "config.json").exists():
            raise FileNotFoundError(
                f"Model directory not found or missing config.json: {model_path}. "
                f"Run `./tools/fetch_models.sh <hf_id> {model_path.name}` to populate it."
            )

        template_path = Path(__file__).parent / self._prompt_file
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template file not found: {template_path}")
        self._prompt_template = template_path.read_text()

        from vllm import LLM, SamplingParams

        self._logger.info(f"Loading vLLM model from {model_path} (dtype={self._dtype})")
        self._llm = LLM(
            model=str(model_path),
            dtype=self._dtype,
            gpu_memory_utilization=self._gpu_memory_utilization,
            max_model_len=self._max_model_len,
            enforce_eager=self._enforce_eager,
        )
        self._sampling_params = SamplingParams(
            temperature=self._temperature,
            top_p=self._top_p,
            max_tokens=self._max_tokens,
            repetition_penalty=self._repetition_penalty,
        )

        try:
            self._llm.chat(
                [{"role": "user", "content": "Say hello."}],
                sampling_params=SamplingParams(max_tokens=5),
                use_tqdm=False,
            )
        except Exception as e:
            self._logger.error(f"vLLM warmup chat failed: {e}")

    def set_on_config(self, prop: str, value: typing.Any): pass

    def start(self): super().start()

    def stop(self):
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

    def _generate(self, prompt: str) -> str:
        outputs = self._llm.chat(
            [{"role": "user", "content": prompt}],
            sampling_params=self._sampling_params,
            use_tqdm=False,
        )
        return outputs[0].outputs[0].text

    def update(self, message: Message[ObjectPayload]):
        transcript = message.payload.get("full_transcript", "")
        trigger_time = message.payload.get("trigger_time", time.time())

        prompt = self._prompt_template.format(transcript=transcript)

        try:
            content = self._generate(prompt).strip()
            content = re.sub(r"<\|[^|]+\|>", "", content)
            content = re.sub(r"^```[a-z]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
            content = content.strip()
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
