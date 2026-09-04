from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.document.service import DocumentService
from app.rag.citations import build_citations
from app.rag.context import Context, ContextBuilder
from app.rag.query import QueryProcessor
from app.retrieval.models import RetrievalResult
from app.services.llm_service import AnswerResult, LLMService


@dataclass
class PipelineResult:
    rewritten_query: str
    entities: list[dict[str, Any]]
    intent: dict[str, Any]
    candidates: list[RetrievalResult]
    context: Context | None = None
    citations: list[dict[str, Any]] | None = None
    answer: AnswerResult | None = None
    graph_fallback: bool = False


class RAGPipeline:
    def __init__(
        self,
        settings: Any,
        document_service: DocumentService,
        entity_service: Any,
        intent_service: Any,
        llm_service: LLMService,
    ) -> None:
        self.settings = settings
        self.document_service = document_service
        self.entity_service = entity_service
        self.intent_service = intent_service
        self.llm_service = llm_service
        self.query_processor = QueryProcessor()
        self.context_builder = ContextBuilder(getattr(settings, "max_context_tokens", 6000))

    def run(
        self,
        query: str,
        history: list[dict[str, Any]],
        access_levels: list[str] | None = None,
        department: str | None = None,
    ) -> PipelineResult:
        normalized_query = self.query_processor.normalize(query)
        rewritten_query = self.query_processor.rewrite(normalized_query, history)
        entities = self.entity_service.extract(rewritten_query)
        intent = self.intent_service.classify(rewritten_query, entities)
        candidates = self.document_service.search(
            rewritten_query,
            top_k=max(20, getattr(self.settings, "retrieval_top_k", 5) * 4),
            access_levels=set(access_levels or []),
            department=department,
            rerank=True,
        )
        if self.document_service.has_relevant_user_result(candidates):
            document_candidates = [item for item in candidates if item.chunk.source == "document"]
            context = self.context_builder.build(document_candidates)
            citations = build_citations(context.results)
            answer = self.llm_service.generate_with_context(normalized_query, context.text, citations)
            return PipelineResult(
                rewritten_query=rewritten_query,
                entities=entities,
                intent=intent,
                candidates=candidates,
                context=context,
                citations=citations,
                answer=answer,
            )
        if not entities:
            return PipelineResult(
                rewritten_query=rewritten_query,
                entities=[],
                intent=intent,
                candidates=candidates,
                citations=[],
                answer=AnswerResult(
                    answer="当前知识库中没有检索到足够可靠的相关信息，因此无法给出确定答案。",
                    grounded=False,
                    model="no-answer",
                ),
            )
        return PipelineResult(
            rewritten_query=rewritten_query,
            entities=entities,
            intent=intent,
            candidates=candidates,
            graph_fallback=True,
        )
