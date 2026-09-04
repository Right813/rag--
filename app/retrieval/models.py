from dataclasses import dataclass
from typing import Any

from app.document.models import Chunk


@dataclass
class RetrievalResult:
    chunk: Chunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    fusion_score: float = 0.0
    reranker_score: float = 0.0
    rank: int = 0

    @property
    def score(self) -> float:
        return self.reranker_score or self.fusion_score or max(self.dense_score, self.bm25_score)

    def to_dict(self) -> dict[str, Any]:
        data = self.chunk.to_dict()
        data.update(
            {
                "score": round(self.score, 6),
                "dense_score": round(self.dense_score, 6),
                "bm25_score": round(self.bm25_score, 6),
                "fusion_score": round(self.fusion_score, 6),
                "reranker_score": round(self.reranker_score, 6),
                "rank": self.rank,
            }
        )
        return data
