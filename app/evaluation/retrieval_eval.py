from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.evaluation.metrics import evaluate_ranking


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    relevant_ids: frozenset[str]


def load_dataset(path: str | Path) -> list[EvaluationCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("questions", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("评测集顶层必须是数组，或包含 questions 数组的对象")
    cases = []
    for row in rows:
        if not isinstance(row, dict) or not str(row.get("query", "")).strip():
            raise ValueError("每条评测样本都必须包含 query")
        relevant_values: list[Any] = []
        for field_name in ("relevant_ids", "relevant_chunk_ids", "relevant_document_ids"):
            values = row.get(field_name, [])
            if isinstance(values, str):
                relevant_values.append(values)
            elif isinstance(values, list):
                relevant_values.extend(values)
        if not relevant_values:
            raise ValueError(f"评测问题缺少相关 ID：{row['query']}")
        cases.append(
            EvaluationCase(
                query=str(row["query"]).strip(),
                relevant_ids=frozenset(str(value) for value in relevant_values),
            )
        )
    return cases


def evaluate_retrieval(
    search: Callable[[str, int], Iterable[Any]],
    cases: list[EvaluationCase],
    ks: tuple[int, ...] = (5, 10),
    candidate_k: int | None = None,
) -> dict[str, float | int]:
    rankings: list[list[str]] = []
    relevant_sets = [case.relevant_ids for case in cases]
    requested_k = candidate_k or max(ks)
    for case in cases:
        results = search(case.query, requested_k)
        rankings.append([_result_ids(result) for result in results])
    metrics = evaluate_ranking(rankings, relevant_sets, ks)
    return {"queries": len(cases), **metrics}


def _result_ids(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("chunk_id") or result.get("document_id") or "")
    chunk = getattr(result, "chunk", None)
    return str(getattr(chunk, "chunk_id", "") or getattr(chunk, "document_id", ""))
