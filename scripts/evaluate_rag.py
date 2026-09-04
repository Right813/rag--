import argparse
import json
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.evaluation.generation import evaluate_generation
from app.evaluation.retrieval_eval import load_dataset
from app.main import create_app
from app.services.document_qa import get_document_service


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 RAG 检索和生成质量评测")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "data/evaluation/sample_questions.json"))
    args = parser.parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    raw_payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    raw_rows = raw_payload.get("questions", []) if isinstance(raw_payload, dict) else raw_payload
    cases = load_dataset(dataset_path)
    application = create_app(get_settings())
    application.state.repository.initialize()
    service = get_document_service(application.state.qa_service)
    service.initialize()
    generation_rows = []
    for index, case in enumerate(cases):
        response = application.state.qa_service.chat(case.query, session_id=f"eval-{uuid.uuid4().hex[:12]}")
        source = raw_rows[index] if index < len(raw_rows) and isinstance(raw_rows[index], dict) else {}
        generation_rows.append(
            {
                "query": case.query,
                "answer": response.answer,
                "grounded": response.grounded,
                "no_answer": response.no_answer,
                "evidence": [item.model_dump() for item in response.evidence],
                "citations": [item.model_dump() for item in response.citations],
                "expected_contains": source.get("expected_contains", []),
                "expected_citation_ids": source.get("expected_citation_ids", []),
                "expected_no_answer": source.get("expected_no_answer"),
            }
        )
    print(json.dumps({"generation": evaluate_generation(generation_rows)}, ensure_ascii=False, indent=2))
    service.close()


if __name__ == "__main__":
    main()
