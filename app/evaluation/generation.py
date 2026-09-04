from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from app.retrieval.tokenizer import tokenize


def evaluate_generation(rows: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    samples = list(rows)
    if not samples:
        return {
            "samples": 0,
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "answer_completeness": 0.0,
            "citation_accuracy": 0.0,
            "no_answer_accuracy": 0.0,
        }
    scores = {
        "faithfulness": [],
        "answer_relevance": [],
        "answer_completeness": [],
        "citation_accuracy": [],
        "no_answer_accuracy": [],
    }
    for row in samples:
        answer = str(row.get("answer") or "")
        evidence = row.get("evidence") or []
        citations = row.get("citations") or []
        query = str(row.get("query") or "")
        expected_contains = [str(item) for item in row.get("expected_contains", []) if str(item)]
        expected_no_answer = row.get("expected_no_answer")
        scores["faithfulness"].append(_faithfulness(answer, evidence, bool(row.get("grounded"))))
        scores["answer_relevance"].append(_answer_relevance(query, answer, bool(expected_no_answer)))
        scores["answer_completeness"].append(_completeness(answer, expected_contains))
        scores["citation_accuracy"].append(_citation_accuracy(citations, evidence, row))
        if expected_no_answer is None:
            scores["no_answer_accuracy"].append(1.0)
        else:
            actual_no_answer = bool(row.get("no_answer")) or not bool(row.get("grounded"))
            scores["no_answer_accuracy"].append(1.0 if actual_no_answer == bool(expected_no_answer) else 0.0)
    return {
        "samples": len(samples),
        **{
            name: round(sum(values) / len(values), 6)
            for name, values in scores.items()
        },
    }


def _faithfulness(answer: str, evidence: list[Any], grounded: bool) -> float:
    if not evidence:
        return 1.0 if not grounded or _looks_like_no_answer(answer) else 0.0
    evidence_text = " ".join(_evidence_text(item) for item in evidence).casefold()
    answer_tokens = set(tokenize(answer))
    if not answer_tokens:
        return 0.0
    supported_tokens = answer_tokens & set(tokenize(evidence_text))
    return min(1.0, len(supported_tokens) / len(answer_tokens))


def _answer_relevance(query: str, answer: str, expected_no_answer: bool) -> float:
    if expected_no_answer:
        return 1.0 if _looks_like_no_answer(answer) else 0.0
    query_tokens = set(tokenize(query))
    answer_tokens = set(tokenize(answer))
    if not query_tokens or not answer_tokens:
        return 0.0
    return min(1.0, len(query_tokens & answer_tokens) / len(query_tokens))


def _completeness(answer: str, expected_contains: list[str]) -> float:
    if not expected_contains:
        return 1.0
    answer_value = answer.casefold()
    return sum(item.casefold() in answer_value for item in expected_contains) / len(expected_contains)


def _citation_accuracy(citations: list[Any], evidence: list[Any], row: dict[str, Any]) -> float:
    expected_ids = {str(item) for item in row.get("expected_citation_ids", []) if str(item)}
    if expected_ids:
        actual_ids = {
            str(item.get("document_id") or item.get("chunk_id") or "")
            for item in citations
            if isinstance(item, dict)
        }
        return len(actual_ids & expected_ids) / len(expected_ids)
    if not citations:
        return 1.0 if not evidence else 0.0
    evidence_ids = {
        str(item.get("document_id") or item.get("chunk_id") or "")
        for item in evidence
        if isinstance(item, dict)
    }
    actual_ids = {
        str(item.get("document_id") or item.get("chunk_id") or "")
        for item in citations
        if isinstance(item, dict)
    }
    if not actual_ids:
        return 0.0
    return len(actual_ids & evidence_ids) / len(actual_ids)


def _evidence_text(item: Any) -> str:
    if isinstance(item, dict):
        return " ".join(
            str(item.get(field) or "")
            for field in ("entity", "relation", "value", "description", "content")
        )
    return str(item)


def _looks_like_no_answer(answer: str) -> bool:
    value = answer.casefold()
    return any(
        marker in value
        for marker in ("无法", "没有检索到", "信息不足", "无法回答", "no answer")
    )
