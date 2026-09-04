from app.evaluation.metrics import evaluate_ranking, ndcg_at_k, recall_at_k
from app.evaluation.retrieval_eval import EvaluationCase, evaluate_retrieval, load_dataset


def test_ranking_metrics_use_relevant_ids_and_cutoff():
    ranking = ["wrong", "right-a", "right-b"]
    relevant = {"right-a", "right-b"}

    assert recall_at_k(ranking, relevant, 1) == 0
    assert recall_at_k(ranking, relevant, 3) == 1
    assert 0 < ndcg_at_k(ranking, relevant, 3) < 1
    assert evaluate_ranking([ranking], [relevant]) == {
        "recall@5": 1.0,
        "recall@10": 1.0,
        "ndcg@5": 0.693426,
        "ndcg@10": 0.693426,
        "mrr": 0.5,
    }


def test_retrieval_evaluation_loads_dataset_and_scores_results(tmp_path):
    dataset_path = tmp_path / "questions.json"
    dataset_path.write_text(
        '{"questions": [{"query": "编号", "rewritten_query": "文档编号", "relevant_chunk_ids": ["chunk-1"]}]}',
        encoding="utf-8",
    )
    cases = load_dataset(dataset_path)
    assert cases == [EvaluationCase(query="编号", relevant_ids=frozenset({"chunk-1"}), rewritten_query="文档编号")]

    metrics = evaluate_retrieval(
        lambda query, top_k: [{"chunk_id": "chunk-1"}],
        cases,
    )
    assert metrics["queries"] == 1
    assert metrics["recall@5"] == 1.0
    assert metrics["mrr"] == 1.0
