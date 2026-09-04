from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from app.document.models import Chunk
from app.retrieval.fusion import rrf_fusion, weighted_fusion
from app.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)


class MilvusVectorStore:
    output_fields = [
        "id",
        "document_id",
        "document_version",
        "content",
        "title",
        "department",
        "category",
        "source",
        "page",
        "chapter",
        "section",
        "chunk_index",
        "created_at",
        "updated_at",
        "status",
        "access_level",
    ]

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.enabled = bool(getattr(settings, "milvus_enabled", True))
        self.uri = str(getattr(settings, "milvus_uri", "http://127.0.0.1:19530"))
        self.token = str(getattr(settings, "milvus_token", "") or "")
        self.db_name = str(getattr(settings, "milvus_db_name", "default") or "default")
        self.collection_name = str(
            getattr(settings, "milvus_collection", "medical_knowledge_chunks")
        )
        self.dimension = int(getattr(settings, "embedding_dimension", 1024))
        self.timeout = float(getattr(settings, "milvus_timeout_seconds", 10.0))
        self.client: Any | None = None
        self.available = False
        self.status = "disabled" if not self.enabled else "unavailable"
        self.chunks: dict[str, Chunk] = {}
        self._fingerprints: dict[str, str] = {}

    def initialize(self, dimension: int | None = None) -> bool:
        if dimension:
            self.dimension = int(dimension)
        if not self.enabled:
            self.status = "disabled"
            return False
        if self.available and self.client is not None:
            return True
        try:
            from pymilvus import MilvusClient

            client_options: dict[str, Any] = {"uri": self.uri}
            if self.token:
                client_options["token"] = self.token
            if self.db_name:
                client_options["db_name"] = self.db_name
            self.client = MilvusClient(**client_options)
            self._ensure_collection()
            self.available = True
            self.status = "ready"
            return True
        except Exception as exc:
            self.client = None
            self.available = False
            self.status = "fallback"
            logger.warning("Milvus unavailable, using local retrieval: %s", exc)
            return False

    def sync(
        self,
        chunks: list[Chunk],
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict[int, float]],
    ) -> bool:
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        if not self.initialize(len(dense_vectors[0]) if dense_vectors else self.dimension):
            return False
        if self.client is None:
            return False
        current_ids = set(self.chunks)
        existing_ids = self._query_ids()
        stale_ids = existing_ids - current_ids
        for group in _groups(sorted(stale_ids), 500):
            self.client.delete(
                collection_name=self.collection_name,
                filter=_in_filter("id", group),
            )
        rows = []
        current_fingerprints: dict[str, str] = {}
        for chunk, dense_vector, sparse_vector in zip(chunks, dense_vectors, sparse_vectors):
            fingerprint = self._fingerprint(chunk, dense_vector, sparse_vector)
            current_fingerprints[chunk.chunk_id] = fingerprint
            if self._fingerprints.get(chunk.chunk_id) == fingerprint and chunk.chunk_id in existing_ids:
                continue
            rows.append(self._row(chunk, dense_vector, sparse_vector))
        if rows:
            upsert = getattr(self.client, "upsert", None)
            if upsert is None:
                self.client.insert(collection_name=self.collection_name, data=rows)
            else:
                upsert(collection_name=self.collection_name, data=rows)
        self._fingerprints = current_fingerprints
        self._load_collection()
        return True

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        top_k: int,
        strategy: str = "rrf",
        alpha: float = 0.7,
        filter_expression: str = "",
    ) -> list[RetrievalResult]:
        if not self.available or self.client is None or not self.chunks:
            return []
        try:
            results = self._server_hybrid_search(
                dense_vector,
                sparse_vector,
                top_k,
                strategy,
                alpha,
                filter_expression,
            )
            if results:
                return results
        except Exception as exc:
            logger.info("Milvus hybrid search unavailable, using two-way fusion: %s", exc)
        dense_results = self._field_search(
            "dense_vector",
            dense_vector,
            top_k,
            "COSINE",
            filter_expression,
            "dense_score",
        )
        sparse_results = self._field_search(
            "sparse_vector",
            sparse_vector,
            top_k,
            "IP",
            filter_expression,
            "bm25_score",
        )
        if strategy.casefold() == "weighted":
            return weighted_fusion(dense_results, sparse_results, alpha, top_k)
        return rrf_fusion(dense_results, sparse_results, top_k)

    def dense_search(
        self,
        dense_vector: list[float],
        top_k: int,
        filter_expression: str = "",
    ) -> list[RetrievalResult]:
        if not self.available or self.client is None:
            return []
        return self._field_search(
            "dense_vector",
            dense_vector,
            top_k,
            "COSINE",
            filter_expression,
            "dense_score",
        )

    def sparse_search(
        self,
        sparse_vector: dict[int, float],
        top_k: int,
        filter_expression: str = "",
    ) -> list[RetrievalResult]:
        if not self.available or self.client is None:
            return []
        return self._field_search(
            "sparse_vector",
            sparse_vector,
            top_k,
            "IP",
            filter_expression,
            "bm25_score",
        )

    def close(self) -> None:
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                logger.debug("Milvus client close failed", exc_info=True)
        self.client = None
        self.available = False
        if self.enabled:
            self.status = "unavailable"

    def _server_hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        top_k: int,
        strategy: str,
        alpha: float,
        filter_expression: str,
    ) -> list[RetrievalResult]:
        from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

        request_options: dict[str, Any] = {}
        if filter_expression:
            request_options["expr"] = filter_expression
        dense_request = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {}},
            limit=top_k,
            **request_options,
        )
        sparse_request = AnnSearchRequest(
            data=[sparse_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=top_k,
            **request_options,
        )
        ranker = WeightedRanker(alpha, 1 - alpha) if strategy.casefold() == "weighted" else RRFRanker()
        raw_results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=top_k,
            output_fields=self.output_fields,
        )
        hits = _flatten_hits(raw_results)
        results: list[RetrievalResult] = []
        for rank, hit in enumerate(hits, start=1):
            result = self._result_from_hit(hit, "fusion_score")
            if result is not None:
                result.rank = rank
                results.append(result)
        return results

    def _field_search(
        self,
        field_name: str,
        vector: list[float] | dict[int, float],
        top_k: int,
        metric_type: str,
        filter_expression: str,
        score_name: str,
    ) -> list[RetrievalResult]:
        if not vector:
            return []
        raw_results = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            anns_field=field_name,
            limit=top_k,
            filter=filter_expression,
            output_fields=self.output_fields,
            search_params={"metric_type": metric_type, "params": {}},
        )
        results: list[RetrievalResult] = []
        for hit in _flatten_hits(raw_results):
            result = self._result_from_hit(hit, score_name)
            if result is not None:
                results.append(result)
        return results

    def _result_from_hit(self, hit: Any, score_name: str) -> RetrievalResult | None:
        if not isinstance(hit, dict):
            return None
        entity = hit.get("entity") if isinstance(hit.get("entity"), dict) else hit
        chunk_id = str(hit.get("id") or hit.get("pk") or entity.get("id") or "")
        chunk = self.chunks.get(chunk_id) or _chunk_from_entity(chunk_id, entity)
        if chunk is None:
            return None
        score = float(hit.get("distance", hit.get("score", 0.0)) or 0.0)
        result = RetrievalResult(chunk=chunk)
        setattr(result, score_name, score)
        if score_name == "fusion_score":
            result.reranker_score = 0.0
        return result

    def _ensure_collection(self) -> None:
        if self.client is None:
            return
        if self.client.has_collection(collection_name=self.collection_name):
            existing_dimension = self._existing_dimension()
            if existing_dimension == self.dimension and self._has_required_indexes():
                return
            self.client.drop_collection(collection_name=self.collection_name)
            self._fingerprints.clear()
        from pymilvus import DataType

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="document_version", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="department", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="page", datatype=DataType.INT64)
        schema.add_field(field_name="chapter", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
        schema.add_field(field_name="created_at", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="updated_at", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="status", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="access_level", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.dimension,
        )
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="dense_vector", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def _existing_dimension(self) -> int | None:
        if self.client is None:
            return None
        description = self.client.describe_collection(collection_name=self.collection_name)
        for field in description.get("fields", []):
            if field.get("fieldName") == "dense_vector" or field.get("name") == "dense_vector":
                params = field.get("params", {})
                return int(params["dim"]) if params.get("dim") else None
        return None

    def _has_required_indexes(self) -> bool:
        if self.client is None:
            return False
        try:
            indexes = self.client.list_indexes(collection_name=self.collection_name)
            names = {str(index) for index in indexes}
            return {"dense_vector", "sparse_vector"}.issubset(names)
        except Exception:
            return False

    def _load_collection(self) -> None:
        try:
            self.client.load_collection(collection_name=self.collection_name)
        except Exception:
            logger.debug("Milvus collection load failed", exc_info=True)

    def _query_ids(self) -> set[str]:
        if self.client is None:
            return set()
        try:
            rows = self.client.query(
                collection_name=self.collection_name,
                filter='id != ""',
                output_fields=["id"],
                limit=16384,
            )
            return {str(row.get("id")) for row in rows if row.get("id") is not None}
        except Exception:
            logger.debug("Could not query existing Milvus IDs", exc_info=True)
            return set()

    @staticmethod
    def _row(
        chunk: Chunk,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
    ) -> dict[str, Any]:
        return {
            "id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_version": chunk.version,
            "content": chunk.content[:65535],
            "title": (chunk.title or chunk.filename)[:512],
            "department": chunk.department[:128],
            "category": (chunk.category or "medical_document")[:128],
            "source": chunk.source[:64],
            "page": chunk.page if chunk.page is not None else -1,
            "chapter": chunk.chapter[:512],
            "section": chunk.section[:512],
            "chunk_index": int(chunk.chunk_index),
            "created_at": chunk.created_at[:64],
            "updated_at": chunk.updated_at[:64],
            "status": (chunk.status or "active")[:32],
            "access_level": (chunk.access_level or "internal")[:64],
            "dense_vector": dense_vector,
            "sparse_vector": sparse_vector,
        }

    @staticmethod
    def _fingerprint(
        chunk: Chunk,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
    ) -> str:
        payload = {
            "chunk": chunk.to_dict(),
            "dense": dense_vector,
            "sparse": sparse_vector,
        }
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _chunk_from_entity(chunk_id: str, entity: dict[str, Any]) -> Chunk | None:
    content = str(entity.get("content") or "")
    document_id = str(entity.get("document_id") or "")
    if not chunk_id or not document_id or not content:
        return None
    page = entity.get("page")
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        filename=str(entity.get("title") or ""),
        version=str(entity.get("document_version") or "v1"),
        file_type="",
        title=str(entity.get("title") or ""),
        category=str(entity.get("category") or "medical_document"),
        page=None if page in {None, -1} else int(page),
        chapter=str(entity.get("chapter") or ""),
        section=str(entity.get("section") or ""),
        chunk_index=int(entity.get("chunk_index") or 0),
        department=str(entity.get("department") or ""),
        access_level=str(entity.get("access_level") or "internal"),
        source=str(entity.get("source") or "document"),
        created_at=str(entity.get("created_at") or ""),
        updated_at=str(entity.get("updated_at") or ""),
        status=str(entity.get("status") or "active"),
    )


def _flatten_hits(raw_results: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_results, list):
        return []
    if raw_results and isinstance(raw_results[0], list):
        return [item for item in raw_results[0] if isinstance(item, dict)]
    return [item for item in raw_results if isinstance(item, dict)]


def _in_filter(field_name: str, values: list[str]) -> str:
    quoted = ", ".join(json.dumps(value, ensure_ascii=False) for value in values)
    return f"{field_name} in [{quoted}]"


def _groups(values: list[str], size: int) -> list[list[str]]:
    return [values[start : start + size] for start in range(0, len(values), size)]
