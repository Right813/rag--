import math
from collections.abc import Iterable, Sequence


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = {str(item) for item in relevant_ids}
    if not relevant or k <= 0:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    relevant = {str(item) for item in relevant_ids}
    if not relevant:
        return 0.0
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant:
            return 1 / rank
    return 0.0


def mrr(rankings: Sequence[Sequence[str]], relevant_sets: Sequence[Iterable[str]]) -> float:
    if not rankings:
        return 0.0
    scores = [
        reciprocal_rank(retrieved_ids, relevant_ids)
        for retrieved_ids, relevant_ids in zip(rankings, relevant_sets)
    ]
    return sum(scores) / len(scores)


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = {str(item) for item in relevant_ids}
    if not relevant or k <= 0:
        return 0.0
    discounted_gain = sum(
        1 / math.log2(rank + 2)
        for rank, item_id in enumerate(retrieved_ids[:k])
        if item_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    ideal_gain = sum(1 / math.log2(rank + 2) for rank in range(ideal_hits))
    return discounted_gain / ideal_gain if ideal_gain else 0.0


def evaluate_ranking(
    rankings: Sequence[Sequence[str]],
    relevant_sets: Sequence[Iterable[str]],
    ks: Sequence[int] = (5, 10),
) -> dict[str, float]:
    if not rankings:
        return {f"recall@{k}": 0.0 for k in ks} | {f"ndcg@{k}": 0.0 for k in ks} | {"mrr": 0.0}
    result = {
        f"recall@{k}": sum(
            recall_at_k(ranking, relevant, k)
            for ranking, relevant in zip(rankings, relevant_sets)
        ) / len(rankings)
        for k in ks
    }
    result.update(
        {
            f"ndcg@{k}": sum(
                ndcg_at_k(ranking, relevant, k)
                for ranking, relevant in zip(rankings, relevant_sets)
            ) / len(rankings)
            for k in ks
        }
    )
    result["mrr"] = mrr(rankings, relevant_sets)
    return {key: round(value, 6) for key, value in result.items()}
