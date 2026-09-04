from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from app.evaluation.metrics import evaluate_ranking
from app.evaluation.retrieval_eval import EvaluationCase, _result_ids


def evaluate_ablation(
    search: Callable[[str, int, str, bool], Iterable[Any]],
    cases: list[EvaluationCase],
    rewrite_query: Callable[[str], str] | None = None,
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, dict[str, float | int]]:
    variants = {
        "dense": ("dense", False, False),
        "bm25": ("bm25", False, False),
        "hybrid": ("hybrid", False, False),
        "hybrid_reranker": ("hybrid", True, False),
        "hybrid_reranker_rewrite": ("hybrid", True, True),
    }
    result: dict[str, dict[str, float | int]] = {}
    requested_k = max(ks)
    for name, (mode, rerank, rewrite) in variants.items():
        rankings = []
        for case in cases:
            query = rewrite_query(case.query) if rewrite and rewrite_query else case.query
            rows = search(query, requested_k, mode, rerank)
            rankings.append([_result_ids(row) for row in rows])
        result[name] = {"queries": len(cases), **evaluate_ranking(rankings, [case.relevant_ids for case in cases], ks)}
    return result
