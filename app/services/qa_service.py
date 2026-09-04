import hashlib
import logging
import threading
import time
import uuid

from app.core.config import Settings
from app.db.mysql import Database
from app.db.redis import Cache
from app.kg.repository import KnowledgeRepository
from app.schemas.chat import ChatResponse, DocumentEvidence, Entity, IntentInfo
from app.services.entity_service import EntityService
from app.services.intent_service import IntentService
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class QAService:
    def __init__(
        self,
        settings: Settings,
        repository: KnowledgeRepository,
        database: Database,
        cache: Cache,
        entity_service: EntityService,
        intent_service: IntentService,
        llm_service: LLMService,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.database = database
        self.cache = cache
        self.entity_service = entity_service
        self.intent_service = intent_service
        self.llm_service = llm_service
        self.total_requests = 0
        self.total_latency_ms = 0.0
        self._metrics_lock = threading.Lock()

    def chat(self, query: str, session_id: str | None = None) -> ChatResponse:
        cleaned_query = " ".join(query.strip().split())
        if not cleaned_query:
            raise ValueError("问题不能为空")
        if len(cleaned_query) > self.settings.max_query_length:
            raise ValueError(f"问题长度不能超过 {self.settings.max_query_length} 个字符")
        actual_session_id = session_id or uuid.uuid4().hex[:16]
        started_at = time.perf_counter()
        self.database.save_conversation(actual_session_id, cleaned_query[:40])
        self.database.save_message(actual_session_id, "user", cleaned_query)

        entities_data = self.entity_service.extract(cleaned_query)
        intent_data = self.intent_service.classify(cleaned_query, entities_data)
        cache_key = self._cache_key(cleaned_query, entities_data, intent_data)
        cached = self.cache.get_json(cache_key)
        if cached:
            response = ChatResponse.model_validate(cached)
            response.session_id = actual_session_id
            response.cached = True
            response.latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
            self.database.save_message(
                actual_session_id,
                "assistant",
                response.answer,
                intent=response.intent.name,
                entities=[item.model_dump() for item in response.entities],
                evidence=[item.model_dump() for item in response.evidence],
                grounded=response.grounded,
                latency_ms=response.latency_ms,
            )
            self._record_metric(response.latency_ms)
            return response

        evidence_data: list[dict] = []
        for entity in entities_data:
            evidence_data.extend(
                self.repository.retrieve(entity["canonical_name"], intent_data["name"], limit=20)
            )
        evidence_data = self._deduplicate_evidence(evidence_data)
        answer_result = self.llm_service.generate(cleaned_query, intent_data, entities_data, evidence_data)
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        response = ChatResponse(
            answer=answer_result.answer,
            entities=[Entity(**entity) for entity in entities_data],
            intent=IntentInfo(**intent_data),
            evidence=[DocumentEvidence(**item) for item in evidence_data],
            grounded=answer_result.grounded,
            latency_ms=latency_ms,
            session_id=actual_session_id,
            model=answer_result.model,
        )
        cache_payload = response.model_copy(deep=True)
        cache_payload.session_id = "cached"
        self.cache.set_json(cache_key, cache_payload.model_dump())
        message_id = self.database.save_message(
            actual_session_id,
            "assistant",
            response.answer,
            intent=response.intent.name,
            entities=[item.model_dump() for item in response.entities],
            evidence=[item.model_dump() for item in response.evidence],
            grounded=response.grounded,
            latency_ms=response.latency_ms,
        )
        response.message_id = message_id
        self._record_metric(latency_ms)
        return response

    def history(self, session_id: str) -> list[dict]:
        return self.database.get_messages(session_id)

    def metrics(self) -> dict:
        with self._metrics_lock:
            average = self.total_latency_ms / self.total_requests if self.total_requests else 0
            return {"requests": self.total_requests, "average_latency_ms": round(average, 2)}

    def _record_metric(self, latency_ms: float) -> None:
        with self._metrics_lock:
            self.total_requests += 1
            self.total_latency_ms += latency_ms

    @staticmethod
    def _deduplicate_evidence(items: list[dict]) -> list[dict]:
        seen: set[tuple[str, str, str]] = set()
        result = []
        for item in items:
            key = (item["entity"], item["relation"], item["value"])
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result[:20]

    @staticmethod
    def _cache_key(query: str, entities: list[dict], intent: dict) -> str:
        entity_names = ",".join(item.get("canonical_name", item["text"]) for item in entities)
        raw = f"{query}|{entity_names}|{intent['name']}"
        return "qa:v1:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()
from app.services.document_qa import install_document_qa
install_document_qa(QAService)

