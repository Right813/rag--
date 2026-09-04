import hashlib
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from app.document.service import DocumentService
from app.rag.pipeline import RAGPipeline
from app.rag.query import QueryProcessor
from app.schemas.chat import ChatResponse, Entity, IntentInfo
from app.schemas.rag import RetrievalSummary
from app.services.llm_service import AnswerResult

logger = logging.getLogger(__name__)


def install_document_qa(qa_class: type) -> None:
    original_chat = qa_class.chat

    def document_aware_chat(
        self,
        query: str,
        session_id: str | None = None,
        access_levels: list[str] | None = None,
        department: str | None = None,
    ) -> ChatResponse:
        service = _get_document_service(self)
        if service is None or not service.has_active_documents():
            return original_chat(self, query, session_id)
        try:
            return _chat_documents(self, service, query, session_id, access_levels or [], department)
        except Exception:
            logger.exception("Document RAG failed, falling back to graph RAG")
            return original_chat(self, query, session_id)

    qa_class.chat = document_aware_chat


def get_document_service(qa_service: Any) -> DocumentService:
    service = _get_document_service(qa_service)
    if service is None:
        raise RuntimeError("文档服务初始化失败")
    return service


def _get_document_service(qa_service: Any) -> DocumentService | None:
    service = getattr(qa_service, "document_service", None)
    if service is None:
        try:
            service = DocumentService(
                qa_service.settings,
                qa_service.repository,
                Path(__file__).resolve().parents[2],
            )
            service.initialize()
            qa_service.document_service = service
        except Exception as exc:
            logger.warning("Document service unavailable: %s", exc)
            return None
    elif not service.initialized:
        service.initialize()
    return service


def _chat_documents(
    qa_service: Any,
    document_service: DocumentService,
    query: str,
    session_id: str | None,
    access_levels: list[str],
    department: str | None,
) -> ChatResponse:
    cleaned_query = " ".join(query.strip().split())
    if not cleaned_query:
        raise ValueError("问题不能为空")
    if len(cleaned_query) > qa_service.settings.max_query_length:
        raise ValueError(f"问题长度不能超过 {qa_service.settings.max_query_length} 个字符")
    actual_session_id = session_id or uuid.uuid4().hex[:16]
    started_at = time.perf_counter()
    history = qa_service.database.get_messages(actual_session_id, limit=10)
    query_processor = QueryProcessor()
    rewritten_query = query_processor.rewrite(cleaned_query, history)
    qa_service.database.save_conversation(actual_session_id, cleaned_query[:40])
    qa_service.database.save_message(actual_session_id, "user", cleaned_query)
    entities_data = qa_service.entity_service.extract(rewritten_query)
    intent_data = qa_service.intent_service.classify(rewritten_query, entities_data)
    cache_key = _cache_key(rewritten_query, intent_data, access_levels, department)
    cached = qa_service.cache.get_json(cache_key)
    if cached:
        response = ChatResponse.model_validate(cached)
        response.session_id = actual_session_id
        response.cached = True
        response.latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        _save_assistant(qa_service, actual_session_id, response)
        qa_service._record_metric(response.latency_ms)
        return response

    pipeline_result = RAGPipeline(
        qa_service.settings,
        document_service,
        qa_service.entity_service,
        qa_service.intent_service,
        qa_service.llm_service,
    ).run(
        cleaned_query,
        history,
        access_levels=access_levels,
        department=department,
    )
    rewritten_query = pipeline_result.rewritten_query
    entities_data = pipeline_result.entities
    intent_data = pipeline_result.intent
    candidates = pipeline_result.candidates
    if pipeline_result.graph_fallback:
        return _graph_fallback(
            qa_service,
            cleaned_query,
            actual_session_id,
            started_at,
            entities_data,
            intent_data,
            rewritten_query,
        )
    citations = pipeline_result.citations or []
    answer_result = pipeline_result.answer or AnswerResult(
        answer="当前知识库中没有检索到足够可靠的相关信息，因此无法给出确定答案。",
        grounded=False,
        model="no-answer",
    )
    context_results = pipeline_result.context.results if pipeline_result.context else []
    evidence = [_result_to_evidence(item) for item in context_results]
    retrieval = RetrievalSummary(
        dense=len(candidates),
        bm25=len(candidates),
        fusion=len(candidates),
        reranked=len(context_results),
        strategy=getattr(qa_service.settings, "fusion_strategy", "rrf"),
        dense_provider=document_service.retriever.dense_provider,
        sparse_provider=document_service.retriever.sparse_provider,
        vector_store=document_service.retriever.vector_store_provider,
        reranker_provider=document_service.retriever.reranker_provider,
    )

    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    response = ChatResponse(
        answer=answer_result.answer,
        entities=[Entity(**entity) for entity in entities_data],
        intent=IntentInfo(**intent_data),
        evidence=evidence,
        grounded=answer_result.grounded,
        latency_ms=latency_ms,
        session_id=actual_session_id,
        model=answer_result.model,
        citations=citations,
        retrieval=retrieval,
        rewritten_query=rewritten_query,
        no_answer=not answer_result.grounded,
    )
    cached_payload = response.model_copy(deep=True)
    cached_payload.session_id = "cached"
    cached_payload.message_id = None
    qa_service.cache.set_json(cache_key, cached_payload.model_dump())
    response.message_id = _save_assistant(qa_service, actual_session_id, response)
    qa_service._record_metric(latency_ms)
    return response


def _graph_fallback(qa_service: Any, query: str, session_id: str, started_at: float, entities: list[dict], intent: dict, rewritten_query: str) -> ChatResponse:
    evidence_data: list[dict] = []
    for entity in entities:
        evidence_data.extend(qa_service.repository.retrieve(entity["canonical_name"], intent["name"], limit=20))
    evidence_data = qa_service._deduplicate_evidence(evidence_data)
    answer_result = qa_service.llm_service.generate(query, intent, entities, evidence_data)
    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    response = ChatResponse(
        answer=answer_result.answer,
        entities=[Entity(**entity) for entity in entities],
        intent=IntentInfo(**intent),
        evidence=evidence_data,
        grounded=answer_result.grounded,
        latency_ms=latency_ms,
        session_id=session_id,
        model=answer_result.model,
        rewritten_query=rewritten_query,
        no_answer=not answer_result.grounded,
    )
    response.message_id = _save_assistant(qa_service, session_id, response)
    qa_service._record_metric(latency_ms)
    return response


def _save_assistant(qa_service: Any, session_id: str, response: ChatResponse) -> int | None:
    return qa_service.database.save_message(
        session_id,
        "assistant",
        response.answer,
        intent=response.intent.name,
        entities=[item.model_dump() for item in response.entities],
        evidence=[item.model_dump() for item in response.evidence],
        grounded=response.grounded,
        latency_ms=response.latency_ms,
    )


def _result_to_evidence(result: Any) -> dict[str, Any]:
    chunk = result.chunk
    return {
        "entity": chunk.filename,
        "relation": chunk.section or chunk.chapter or "文档片段",
        "value": chunk.content[:500],
        "source": chunk.source,
        "description": chunk.content,
        "document_id": chunk.document_id,
        "filename": chunk.filename,
        "version": chunk.version,
        "category": chunk.category,
        "page": chunk.page,
        "chapter": chunk.chapter,
        "section": chunk.section,
        "score": round(result.score, 6),
    }


def _cache_key(
    query: str,
    intent: dict,
    access_levels: list[str] | None = None,
    department: str | None = None,
) -> str:
    levels = ",".join(sorted(set(access_levels or [])))
    raw = f"{query}|{intent['name']}|{levels}|{department or ''}"
    return "qa:doc:v1:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()
