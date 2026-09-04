import re
from dataclasses import dataclass

from app.document.models import Chunk
from app.document.chunker import estimate_tokens
from app.retrieval.models import RetrievalResult


@dataclass
class Context:
    text: str
    results: list[RetrievalResult]
    tokens: int


class ContextBuilder:
    def __init__(self, max_tokens: int = 6000) -> None:
        self.max_tokens = max(500, max_tokens)

    def build(self, results: list[RetrievalResult]) -> Context:
        selected: list[RetrievalResult] = []
        seen: set[str] = set()
        parts: list[str] = []
        used_tokens = 0
        for result in results:
            normalized = re.sub(r"\s+", " ", result.chunk.content).strip()
            fingerprint = normalized[:240].casefold()
            if not normalized or fingerprint in seen:
                continue
            item_tokens = estimate_tokens(normalized)
            if selected and used_tokens + item_tokens > self.max_tokens:
                break
            seen.add(fingerprint)
            selected.append(result)
            used_tokens += item_tokens
            source = self._source_label(result.chunk)
            parts.append(f"[{len(selected)}] {source}\n{normalized}")
        return Context(text="\n\n".join(parts), results=selected, tokens=used_tokens)

    @staticmethod
    def _source_label(chunk: Chunk) -> str:
        fields = [chunk.filename]
        if chunk.page is not None:
            fields.append(f"第{chunk.page}页")
        if chunk.chapter:
            fields.append(chunk.chapter)
        if chunk.section:
            fields.append(chunk.section)
        return " / ".join(item for item in fields if item)
