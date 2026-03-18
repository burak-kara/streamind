import time
import typing
from difflib import SequenceMatcher
from juturna.components import Node, Message
from juturna.payloads import ObjectPayload


class NovelExtractor(Node[ObjectPayload, ObjectPayload]):
    """Deduplicates overlapping ASR transcriptions using word-level matching."""

    def __init__(self, max_overlap_words: int = 10, similarity_threshold: float = 0.8,
                 **kwargs):
        super().__init__(**kwargs)
        self._max_overlap_words = max_overlap_words
        self._similarity_threshold = similarity_threshold
        self._prev_words: list[str] = []

    def configure(self): pass
    def warmup(self):
        self._prev_words = []
    def set_on_config(self, prop: str, value: typing.Any): pass
    def start(self): super().start()
    def stop(self): super().stop()
    def destroy(self): pass

    def _find_overlap(self, prev_words: list[str], curr_words: list[str]) -> int:
        if not prev_words or not curr_words:
            return 0
        max_check = min(len(prev_words), len(curr_words), self._max_overlap_words)
        for overlap_len in range(max_check, 0, -1):
            prev_suffix = prev_words[-overlap_len:]
            curr_prefix = curr_words[:overlap_len]
            if prev_suffix == curr_prefix:
                return overlap_len
            prev_str = " ".join(prev_suffix)
            curr_str = " ".join(curr_prefix)
            ratio = SequenceMatcher(None, prev_str, curr_str).ratio()
            if ratio >= self._similarity_threshold:
                return overlap_len
        return 0

    def update(self, message: Message[ObjectPayload]):
        transcript = message.payload.get("transcript", "")
        curr_words = transcript.split() if transcript.strip() else []
        overlap = self._find_overlap(self._prev_words, curr_words)
        novel_words = curr_words[overlap:]
        novel_text = " ".join(novel_words)
        self._prev_words = curr_words

        payload = ObjectPayload.from_dict({
            "novel_text": novel_text,
            "chunk_end": message.payload.get("chunk_end", 0.0),
            "wall_time": time.time(),
        })
        out = Message[ObjectPayload](
            creator=self.name,
            version=message.version,
            payload=payload,
            timers_from=message,
        )
        self.transmit(out)
