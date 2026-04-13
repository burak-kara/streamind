import re
import typing
from juturna.components import Node, Message
from juturna.payloads import ObjectPayload


# Inline noise tags to strip from within transcripts before evaluation.
_STRIP_PATTERNS = [
    re.compile(r'\[.*?\]', re.IGNORECASE),  # e.g. [Music], [Applause]
    re.compile(r'\(.*?\)', re.IGNORECASE),  # e.g. (music), (laughter)
]


def _clean(text: str) -> str:
    for pattern in _STRIP_PATTERNS:
        text = pattern.sub('', text)
    return ' '.join(text.split())  # collapse whitespace


class HallucinationFilter(Node[ObjectPayload, ObjectPayload]):
    """Drops or strips Whisper hallucination patterns from transcripts.

    Known exact-match hallucinations are loaded from config.toml at startup
    via the ``exact_hallucinations`` argument. Additional patterns can be
    supplied per-deployment via ``extra_patterns``.

    Messages whose transcript reduces to a known hallucination phrase (or to
    empty) after inline-tag removal are silently dropped. Messages with real
    content are forwarded unchanged; messages with partial inline tags are
    forwarded with those tags removed.
    """

    def __init__(
        self,
        exact_hallucinations: list | None = None,
        extra_patterns: list | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        base = frozenset(p.lower().strip() for p in (exact_hallucinations or []))
        extra = frozenset(p.lower().strip() for p in (extra_patterns or []))
        self._hallucinations = base | extra

    def configure(self): pass
    def warmup(self): pass
    def set_on_config(self, prop: str, value: typing.Any): pass
    def start(self): super().start()
    def stop(self): super().stop()
    def destroy(self): pass

    def update(self, message: Message[ObjectPayload]):
        transcript = message.payload.get("transcript", "")
        cleaned = _clean(transcript)

        if not cleaned or cleaned.lower() in self._hallucinations:
            return  # pure hallucination — drop

        if cleaned == transcript:
            self.transmit(message)  # nothing changed — pass through as-is
            return

        # Inline tags were stripped — rebuild with cleaned text
        payload = ObjectPayload.from_dict({
            "transcript": cleaned,
            "chunk_start": message.payload.get("chunk_start", 0.0),
            "chunk_end": message.payload.get("chunk_end", 0.0),
        })
        out = Message[ObjectPayload](
            creator=self.name,
            version=message.version,
            payload=payload,
            timers_from=message,
        )
        self.transmit(out)
