import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.ablation import evaluate_ablation
from app.evaluation.retrieval_eval import load_dataset
from app.main import create_app
from app.core.config import get_settings
from app.services.document_qa import get_document_service


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Dense/BM25/Hybrid/Reranker 消融评测")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "data/evaluation/sample_questions.json"))
    args = parser.parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    cases = load_dataset(dataset_path)
    rewrite_map = {
        case.query: case.rewritten_query
        for case in cases
        if case.rewritten_query
    }
    application = create_app(get_settings())
    application.state.repository.initialize()
    service = get_document_service(application.state.qa_service)
    service.initialize()
    metrics = evaluate_ablation(
        lambda query, top_k, mode, rerank: service.search(
            query,
            top_k=top_k,
            mode=mode,
            rerank=rerank,
        ),
        cases,
        rewrite_query=lambda query: rewrite_map.get(query) or query,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    service.close()


if __name__ == "__main__":
    main()
