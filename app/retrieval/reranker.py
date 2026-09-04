import logging
from typing import Any

from app.retrieval.models import RetrievalResult
from app.retrieval.tokenizer import tokenize

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str = "") -> None:
        self.model_name = model_name.strip()
        self.model: Any | None = None
        self.provider = "lexical"

    def rerank(self, query: str, candidates: list[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]:
        if not candidates:
            return []
        self._load_model()
        if self.model is not None:
            try:
                pairs = [(query, item.chunk.content) for item in candidates]
                scores = self.model.predict(pairs)
                for item, score in zip(candidates, scores):
                    item.reranker_score = float(score)
            except Exception as exc:
                logger.warning("Reranker failed, using lexical fallback: %s", exc)
                self.model = None
                self.provider = "lexical"
        if self.model is None:
            query_tokens = set(tokenize(query))
            query_value = query.casefold().strip()
            for item in candidates:
                content_value = item.chunk.content.casefold()
                content_tokens = set(tokenize(item.chunk.content))
                overlap = len(query_tokens & content_tokens) / max(len(query_tokens), 1)
                phrase = 1.0 if query_value and query_value in content_value else 0.0
                identifier = 1.0 if any(token in content_value for token in query_tokens if len(token) > 3) else 0.0
                item.reranker_score = min(1.0, 0.62 * overlap + 0.28 * phrase + 0.1 * identifier)
        candidates.sort(key=lambda item: (item.reranker_score, item.fusion_score), reverse=True)
        for rank, item in enumerate(candidates[:top_k], start=1):
            item.rank = rank
        return candidates[:top_k]

    def _load_model(self) -> None:
        if not self.model_name or self.model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)
            self.provider = "cross-encoder"
        except Exception as exc:
            logger.info("Cross encoder unavailable, using lexical reranker: %s", exc)
            self.model = None
            self.provider = "lexical"
