from app.core.config import Settings
from app.document.models import Chunk
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import rrf_fusion, weighted_fusion
from app.retrieval.milvus_store import MilvusVectorStore
from app.retrieval.models import RetrievalResult
from app.retrieval.reranker import Reranker


class HybridRetriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dense = DenseRetriever(settings.embedding_model, settings.embedding_dimension)
        self.bm25 = BM25Retriever()
        self.reranker = Reranker(settings.reranker_model)
        self.vector_store = MilvusVectorStore(settings)
        self.chunks: list[Chunk] = []
        self.initialized = False

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self.dense.build(self.chunks)
        self.bm25.build(self.chunks)
        self.vector_store.sync(
            self.chunks,
            self.dense.vectors,
            [self.bm25.sparse_vector(chunk.content) for chunk in self.chunks],
        )
        self.initialized = True

    def search(
        self,
        query: str,
        top_k: int | None = None,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
        strategy: str | None = None,
        rerank: bool = True,
        filter_expression: str = "",
        mode: str = "hybrid",
    ) -> list[RetrievalResult]:
        if not self.initialized:
            return []
        final_top_k = top_k or self.settings.retrieval_top_k
        fusion_strategy = (strategy or self.settings.fusion_strategy).casefold()
        candidate_top_k = max(
            final_top_k * 4,
            dense_top_k or self.settings.dense_top_k,
            bm25_top_k or self.settings.bm25_top_k,
            20,
        )
        retrieval_mode = mode.casefold()
        if retrieval_mode == "dense":
            candidates = (
                self.vector_store.dense_search(
                    self.dense.encode(query), candidate_top_k, filter_expression
                )
                if self.vector_store.available
                else self.dense.search(query, candidate_top_k)
            )
        elif retrieval_mode == "bm25":
            candidates = (
                self.vector_store.sparse_search(
                    self.bm25.query_sparse_vector(query), candidate_top_k, filter_expression
                )
                if self.vector_store.available
                else self.bm25.search(query, candidate_top_k)
            )
        elif self.vector_store.available:
            candidates = self.vector_store.hybrid_search(
                self.dense.encode(query),
                self.bm25.query_sparse_vector(query),
                candidate_top_k,
                fusion_strategy,
                self.settings.fusion_alpha,
                filter_expression,
            )
        else:
            dense_results = self.dense.search(query, dense_top_k or self.settings.dense_top_k)
            sparse_results = self.bm25.search(query, bm25_top_k or self.settings.bm25_top_k)
            if fusion_strategy == "weighted":
                candidates = weighted_fusion(
                    dense_results,
                    sparse_results,
                    self.settings.fusion_alpha,
                    candidate_top_k,
                )
            else:
                candidates = rrf_fusion(dense_results, sparse_results, candidate_top_k)
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

    @property
    def sparse_provider(self) -> str:
        return "milvus-bm25" if self.vector_store.available else "bm25-local"

    @property
    def vector_store_provider(self) -> str:
        return "milvus" if self.vector_store.available else "local"

    @property
    def vector_status(self) -> str:
        return self.vector_store.status

    @property
    def collection_name(self) -> str:
        return self.vector_store.collection_name

    def close(self) -> None:
        self.vector_store.close()
