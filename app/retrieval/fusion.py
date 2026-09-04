from app.retrieval.models import RetrievalResult


def weighted_fusion(
    dense_results: list[RetrievalResult],
    sparse_results: list[RetrievalResult],
    alpha: float = 0.7,
    top_k: int = 30,
) -> list[RetrievalResult]:
    by_chunk: dict[str, RetrievalResult] = {}
    dense_max = max((item.dense_score for item in dense_results), default=0.0)
    sparse_max = max((item.bm25_score for item in sparse_results), default=0.0)
    for item in dense_results:
        by_chunk[item.chunk.chunk_id] = RetrievalResult(
            chunk=item.chunk,
            dense_score=item.dense_score,
            fusion_score=alpha * _dense_normalize(item.dense_score, dense_max),
        )
    for item in sparse_results:
        current = by_chunk.get(item.chunk.chunk_id)
        if current is None:
            current = RetrievalResult(chunk=item.chunk)
            by_chunk[item.chunk.chunk_id] = current
        current.bm25_score = item.bm25_score
        current.fusion_score += (1 - alpha) * _sparse_normalize(item.bm25_score, sparse_max)
    result = sorted(by_chunk.values(), key=lambda item: item.fusion_score, reverse=True)
    return result[:top_k]


def rrf_fusion(
    dense_results: list[RetrievalResult],
    sparse_results: list[RetrievalResult],
    top_k: int = 30,
    rank_constant: int = 60,
) -> list[RetrievalResult]:
    by_chunk: dict[str, RetrievalResult] = {}
    for result_list, score_name in ((dense_results, "dense_score"), (sparse_results, "bm25_score")):
        for rank, item in enumerate(result_list, start=1):
            current = by_chunk.get(item.chunk.chunk_id)
            if current is None:
                current = RetrievalResult(chunk=item.chunk)
                by_chunk[item.chunk.chunk_id] = current
            setattr(current, score_name, getattr(item, score_name))
            current.fusion_score += 1 / (rank_constant + rank)
    result = sorted(by_chunk.values(), key=lambda item: item.fusion_score, reverse=True)
    return result[:top_k]


def _dense_normalize(score: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return max(0.0, min(1.0, (score + 1) / 2))


def _sparse_normalize(score: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return max(0.0, min(1.0, score / maximum))
