from __future__ import annotations
import json
import logging
import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from app.document.chunker import StructureAwareChunker
from app.document.models import Chunk, DocumentRecord, utc_now
from app.document.parsers import DocumentParseError, DocumentParser, SUPPORTED_EXTENSIONS
from app.kg.repository import KnowledgeRepository
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self, settings: Any, repository: KnowledgeRepository, project_root: str | Path = ".") -> None:
        self.settings = settings
        self.repository = repository
        self.project_root = Path(project_root).resolve()
        self.storage_path = self._resolve_path(getattr(settings, "document_storage_path", "data/documents.json"))
        self.upload_dir = self._resolve_path(getattr(settings, "document_upload_dir", "data/uploads"))
        self.max_upload_size = int(getattr(settings, "max_upload_size_bytes", 20 * 1024 * 1024))
        self.parser = DocumentParser(bool(getattr(settings, "ocr_enabled", True)))
        self.chunker = StructureAwareChunker(
            int(getattr(settings, "chunk_size", 600)),
            int(getattr(settings, "chunk_overlap", 80)),
        )
        self.retriever = HybridRetriever(settings)
        self.documents: dict[str, DocumentRecord] = {}
        self.chunks: dict[str, list[Chunk]] = {}
        self._lock = threading.RLock()
        self.initialized = False

    def initialize(self) -> None:
        with self._lock:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.upload_dir.mkdir(parents=True, exist_ok=True)
            self._load()
            self._rebuild_index()
            self.initialized = True

    def upload(
        self,
        filename: str,
        content: bytes,
        department: str = "",
        access_level: str = "internal",
        version: str | None = None,
    ) -> DocumentRecord:
        safe_name = self._safe_filename(filename)
        suffix = Path(safe_name).suffix.casefold()
        self._validate_upload(safe_name, content)
        document_id = uuid.uuid4().hex
        document_version = version.strip() if version and version.strip() else self._next_version(safe_name)
        document_dir = self.upload_dir / document_id
        document_dir.mkdir(parents=True, exist_ok=False)
        raw_path = document_dir / safe_name
        raw_path.write_bytes(content)
        record = DocumentRecord(
            document_id=document_id,
            filename=safe_name,
            version=document_version,
            file_type=suffix.lstrip("."),
            size_bytes=len(content),
            raw_path=str(raw_path),
            department=department.strip()[:100],
            access_level=access_level.strip()[:40] or "internal",
        )
        with self._lock:
            self.documents[document_id] = record
            try:
                self._index_record(record)
            except Exception as exc:
                record.status = "failed"
                record.error = str(exc)[:500]
                record.updated_at = utc_now()
                self._persist()
                self._rebuild_index()
                raise DocumentParseError(record.error) from exc
            self._archive_same_filename(safe_name, exclude_document_id=document_id)
            self._persist()
            self._rebuild_index()
        return record

    def reindex(self, document_id: str) -> DocumentRecord:
        with self._lock:
            record = self._get_record(document_id)
            record.status = "processing"
            record.error = None
            record.updated_at = utc_now()
            try:
                self._index_record(record)
            except Exception as exc:
                record.status = "failed"
                record.error = str(exc)[:500]
                record.updated_at = utc_now()
                self._persist()
                self._rebuild_index()
                raise DocumentParseError(record.error) from exc
            self._persist()
            self._rebuild_index()
            return record

    def delete(self, document_id: str) -> None:
        with self._lock:
            record = self._get_record(document_id)
            self.documents.pop(document_id, None)
            self.chunks.pop(document_id, None)
            raw_path = Path(record.raw_path).resolve()
            if self._is_within(raw_path, self.upload_dir.resolve()) and raw_path.exists():
                shutil.rmtree(raw_path.parent)
            self._persist()
            self._rebuild_index()

    def get(self, document_id: str) -> DocumentRecord:
        with self._lock:
            return self._get_record(document_id)

    def list(self, status: str | None = None, search: str | None = None) -> list[DocumentRecord]:
        query = (search or "").casefold().strip()
        with self._lock:
            records = [
                record
                for record in self.documents.values()
                if (not status or record.status == status)
                and (not query or query in record.filename.casefold() or query in record.department.casefold())
            ]
            return sorted(records, key=lambda item: item.updated_at, reverse=True)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        strategy: str | None = None,
        include_legacy: bool = True,
        access_levels: set[str] | None = None,
        rerank: bool = True,
    ) -> list[RetrievalResult]:
        with self._lock:
            if not self.initialized:
                return []
            requested_top_k = top_k or self.settings.retrieval_top_k
            candidate_top_k = max(requested_top_k * 4, requested_top_k)
            results = self.retriever.search(query, top_k=candidate_top_k, strategy=strategy, rerank=rerank)
            allowed = access_levels or set()
            if not include_legacy:
                results = [item for item in results if item.chunk.source == "document"]
            if allowed:
                results = [item for item in results if item.chunk.access_level in allowed or item.chunk.source == "medical_kg"]
            return results[:requested_top_k]

    def has_active_documents(self) -> bool:
        return any(record.status == "active" for record in self.documents.values())

    def has_relevant_user_result(self, results: list[RetrievalResult]) -> bool:
        threshold = float(getattr(self.settings, "no_answer_threshold", 0.14))
        return any(item.chunk.source == "document" and item.score >= threshold for item in results)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            by_status: dict[str, int] = {}
            for record in self.documents.values():
                by_status[record.status] = by_status.get(record.status, 0) + 1
            return {
                "documents": len(self.documents),
                "active_documents": by_status.get("active", 0),
                "chunks": sum(len(items) for items in self.chunks.values()),
                "by_status": by_status,
                "dense_provider": self.retriever.dense_provider,
                "sparse_provider": "bm25",
                "fusion_strategy": getattr(self.settings, "fusion_strategy", "rrf"),
                "reranker_provider": self.retriever.reranker_provider,
            }

    def _index_record(self, record: DocumentRecord) -> None:
        path = Path(record.raw_path).resolve()
        if not self._is_within(path, self.upload_dir.resolve()) or not path.is_file():
            raise DocumentParseError("文档存储路径无效")
        record.status = "processing"
        pages = self.parser.parse(path)
        parsed_chunks = self.chunker.split(pages, record)
        if not parsed_chunks:
            raise DocumentParseError("文档没有可建立索引的有效内容")
        record.chunk_count = len(parsed_chunks)
        record.status = "active"
        record.error = None
        record.updated_at = utc_now()
        self.chunks[record.document_id] = parsed_chunks

    def _rebuild_index(self) -> None:
        active_chunks = [
            chunk
            for document_id, document_chunks in self.chunks.items()
            if self.documents.get(document_id, DocumentRecord("", "", "", "", 0, "")).status == "active"
            for chunk in document_chunks
        ]
        active_chunks.extend(self._legacy_chunks())
        self.retriever.build(active_chunks)

    def _legacy_chunks(self) -> list[Chunk]:
        target_by_name = {item["name"]: item for item in self.repository.entities}
        chunks = []
        for index, relation in enumerate(self.repository.relations):
            target = target_by_name.get(relation["target"], {})
            relation_label = _relation_label(relation["relation"])
            text = f"{relation['source']}的{relation_label}：{relation['target']}。{target.get('description', '')}"
            chunks.append(
                Chunk(
                    chunk_id=f"legacy-{index}",
                    document_id="legacy-medical-kg",
                    content=text,
                    filename="内置医疗示例知识库",
                    version="v1",
                    file_type="json",
                    chapter="内置知识",
                    section=relation_label,
                    access_level="internal",
                    source="medical_kg",
                    entity=relation["source"],
                    relation=relation["relation"],
                    value=relation["target"],
                )
            )
        return chunks

    def refresh_legacy(self) -> None:
        with self._lock:
            self._rebuild_index()

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
            self.documents = {
                item["document_id"]: DocumentRecord(**item)
                for item in payload.get("documents", [])
                if item.get("document_id")
            }
            self.chunks = {
                document_id: [Chunk(**chunk) for chunk in items]
                for document_id, items in payload.get("chunks", {}).items()
            }
        except (OSError, TypeError, ValueError, KeyError) as exc:
            logger.warning("Could not load document index, starting empty: %s", exc)
            self.documents = {}
            self.chunks = {}

    def _persist(self) -> None:
        payload = {
            "documents": [record.to_dict() for record in self.documents.values()],
            "chunks": {document_id: [chunk.to_dict() for chunk in items] for document_id, items in self.chunks.items()},
        }
        temporary_path = self.storage_path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_path.replace(self.storage_path)

    def _archive_same_filename(self, filename: str, exclude_document_id: str | None = None) -> None:
        for record in self.documents.values():
            if (
                record.document_id != exclude_document_id
                and record.filename.casefold() == filename.casefold()
                and record.status == "active"
            ):
                record.status = "archived"
                record.updated_at = utc_now()

    def _next_version(self, filename: str) -> str:
        versions = []
        for record in self.documents.values():
            if record.filename.casefold() == filename.casefold():
                match = re.search(r"(\d+)$", record.version)
                if match:
                    versions.append(int(match.group(1)))
        return f"v{max(versions, default=0) + 1}"

    def _get_record(self, document_id: str) -> DocumentRecord:
        record = self.documents.get(document_id)
        if record is None:
            raise KeyError(f"文档不存在：{document_id}")
        return record

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path

    @staticmethod
    def _safe_filename(filename: str) -> str:
        value = Path(filename or "").name
        value = re.sub(r"[\x00-\x1f\x7f]", "", value).strip()
        value = re.sub(r"[<>:\"/\\|?*]", "_", value)
        if not value or value in {".", ".."}:
            raise DocumentParseError("文件名无效")
        return value[:180]

    def _validate_upload(self, filename: str, content: bytes) -> None:
        suffix = Path(filename).suffix.casefold()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise DocumentParseError("仅支持 PDF、DOCX、XLSX、TXT、Markdown、JPG 或 PNG")
        if not content:
            raise DocumentParseError("文件不能为空")
        if len(content) > self.max_upload_size:
            raise DocumentParseError(f"文件不能超过 {self.max_upload_size // 1024 // 1024} MB")
        signatures = {
            ".pdf": content.startswith(b"%PDF"),
            ".docx": content.startswith(b"PK"),
            ".xlsx": content.startswith(b"PK"),
            ".png": content.startswith(b"\x89PNG"),
            ".jpg": content.startswith(b"\xff\xd8"),
            ".jpeg": content.startswith(b"\xff\xd8"),
        }
        if suffix in signatures and not signatures[suffix]:
            raise DocumentParseError("文件内容与扩展名不匹配")

    @staticmethod
    def _is_within(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False


def _relation_label(relation: str) -> str:
    return {
        "HAS_SYMPTOM": "常见症状",
        "HAS_DRUG": "相关药物",
        "RECOMMEND_FOOD": "推荐饮食",
        "NEEDS_CHECK": "需要检查",
        "HAS_TREATMENT": "治疗方式",
        "BELONGS_TO_DEPARTMENT": "就诊科室",
        "HAS_EFFECT": "药物作用",
        "HAS_USAGE": "使用方式",
    }.get(relation, relation)
