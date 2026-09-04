from app.core.config import Settings
from app.document.models import Chunk
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import rrf_fusion, weighted_fusion
from app.retrieval.models import RetrievalResult
from app.retrieval.reranker import Reranker


class HybridRetriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dense = DenseRetriever(settings.embedding_model)
        self.bm25 = BM25Retriever()
        self.reranker = Reranker(settings.reranker_model)
        self.chunks: list[Chunk] = []
        self.initialized = False

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self.dense.build(self.chunks)
        self.bm25.build(self.chunks)
        self.initialized = True

    def search(
        self,
        query: str,
        top_k: int | None = None,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
        strategy: str | None = None,
        rerank: bool = True,
    ) -> list[RetrievalResult]:
        if not self.initialized:
            return []
        final_top_k = top_k or self.settings.retrieval_top_k
        dense_results = self.dense.search(query, dense_top_k or self.settings.dense_top_k)
        sparse_results = self.bm25.search(query, bm25_top_k or self.settings.bm25_top_k)
        fusion_strategy = (strategy or self.settings.fusion_strategy).casefold()
        if fusion_strategy == "weighted":
            candidates = weighted_fusion(dense_results, sparse_results, self.settings.fusion_alpha, max(final_top_k * 4, 20))
        else:
            candidates = rrf_fusion(dense_results, sparse_results, max(final_top_k * 4, 20))
        if rerank:
            return self.reranker.rerank(query, candidates, final_top_k)
        for rank, item in enumerate(candidates[:final_top_k], start=1):
            item.rank = rank
            item.reranker_score = item.fusion_score
        return candidates[:final_top_k]

    @property
    def dense_provider(self) -> str:
        return self.dense.provider

    @property
    def reranker_provider(self) -> str:
        return self.reranker.provider
