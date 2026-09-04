import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.evaluation.retrieval_eval import evaluate_retrieval, load_dataset
from app.main import create_app
from app.services.document_qa import get_document_service


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Hybrid RAG 检索离线评测")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "data/evaluation/sample_questions.json"), help="JSON 评测集路径")
    parser.add_argument(
        "--strategy",
        choices=("rrf", "weighted"),
        default=None,
        help="Fusion 策略，默认读取环境配置",
    )
    args = parser.parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    cases = load_dataset(dataset_path)
    application = create_app(get_settings())
    application.state.repository.initialize()
    service = get_document_service(application.state.qa_service)
    service.refresh_legacy()
    metrics = evaluate_retrieval(
        lambda query, top_k: service.search(query, top_k=top_k, strategy=args.strategy, rerank=True),
        cases,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
