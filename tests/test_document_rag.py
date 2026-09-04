from io import BytesIO

from docx import Document
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def _document_client(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        fallback_database_path=str(tmp_path / "fallback.db"),
        redis_enabled=False,
        neo4j_enabled=False,
        milvus_enabled=False,
        embedding_model="",
        llm_base_url="",
        admin_token="",
        knowledge_store_path=str(tmp_path / "knowledge_store.json"),
        document_storage_path=str(tmp_path / "documents.json"),
        document_upload_dir=str(tmp_path / "uploads"),
        chunk_size=220,
        chunk_overlap=30,
    )
    return TestClient(create_app(settings))


def test_document_lifecycle_hybrid_search_citation_and_sse(tmp_path):
    with _document_client(tmp_path) as client:
        content = """第三章 供应商管理\n\n3.1 供应商准入\n\n供应商编号：SUP-2025-00182。注册资本不得低于100万元，采购部负责初审。"""
        uploaded = client.post(
            "/api/v1/documents/upload",
            files={"file": ("供应商制度.txt", content.encode("utf-8"), "text/plain")},
            data={"department": "采购部", "access_level": "internal", "version": "v3"},
        )

        assert uploaded.status_code == 200
        document = uploaded.json()["document"]
        assert document["status"] == "active"
        assert document["chunk_count"] >= 1

        listed = client.get("/api/v1/documents")
        assert listed.status_code == 200
        assert listed.json()["total"] == 1

        search = client.post(
            "/api/v1/retrieval/search",
            json={"query": "SUP-2025-00182", "top_k": 5, "strategy": "rrf"},
        )
        assert search.status_code == 200
        assert search.json()["results"][0]["document_id"] == document["document_id"]

        answer = client.post(
            "/api/v1/chat",
            json={"query": "供应商注册资本最低要求是多少？", "session_id": "doc-test"},
        )
        assert answer.status_code == 200
        assert answer.json()["grounded"] is True
        assert answer.json()["citations"][0]["document_id"] == document["document_id"]

        stream = client.post("/api/v1/chat/stream", json={"query": "供应商编号是什么？"})
        assert stream.status_code == 200
        assert "event: token" in stream.text
        assert "event: done" in stream.text

        reindexed = client.post(f"/api/v1/documents/{document['document_id']}/reindex")
        assert reindexed.status_code == 200
        assert reindexed.json()["document"]["status"] == "active"

        deleted = client.delete(f"/api/v1/documents/{document['document_id']}")
        assert deleted.status_code == 204
        assert client.get("/api/v1/documents").json()["total"] == 0


def test_document_upload_rejects_path_traversal_and_bad_signature(tmp_path):
    with _document_client(tmp_path) as client:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("..\\..\\evil.pdf", b"not-a-pdf", "application/pdf")},
        )

    assert response.status_code == 422


def test_failed_new_version_keeps_previous_active_version(tmp_path):
    with _document_client(tmp_path) as client:
        source_document = Document()
        source_document.add_paragraph("采购制度内容")
        source_buffer = BytesIO()
        source_document.save(source_buffer)
        first = client.post(
            "/api/v1/documents/upload",
            files={"file": ("policy.docx", source_buffer.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            data={"version": "v1"},
        )
        assert first.status_code == 200

        failed = client.post(
            "/api/v1/documents/upload",
            files={"file": ("policy.docx", b"PK\x03\x04", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            data={"version": "v2"},
        )
        assert failed.status_code == 422

        records = client.get("/api/v1/documents").json()["items"]
        assert len(records) == 2
        assert {record["status"] for record in records} == {"active", "failed"}
        assert next(record for record in records if record["version"] == "v1")["status"] == "active"


def test_document_mutations_invalidate_cached_answers(tmp_path):
    with _document_client(tmp_path) as client:
        first = client.post(
            "/api/v1/documents/upload",
            files={"file": ("limits.txt", "注册资本不得低于100万元".encode("utf-8"), "text/plain")},
            data={"version": "v1"},
        )
        assert first.status_code == 200
        first_answer = client.post(
            "/api/v1/chat",
            json={"query": "注册资本最低要求是多少？", "session_id": "cache-v1"},
        )
        assert first_answer.status_code == 200
        assert first_answer.json()["citations"][0]["version"] == "v1"

        second = client.post(
            "/api/v1/documents/upload",
            files={"file": ("limits.txt", "注册资本不得低于300万元".encode("utf-8"), "text/plain")},
            data={"version": "v2"},
        )
        assert second.status_code == 200
        second_answer = client.post(
            "/api/v1/chat",
            json={"query": "注册资本最低要求是多少？", "session_id": "cache-v2"},
        )
        assert second_answer.status_code == 200
        assert second_answer.json()["citations"][0]["version"] == "v2"
        assert "300万元" in second_answer.json()["answer"]


def test_stream_chat_respects_document_access_filters(tmp_path):
    with _document_client(tmp_path) as client:
        uploaded = client.post(
            "/api/v1/documents/upload",
            files={"file": ("restricted.txt", b"SUP-STREAM-001 is restricted procurement guidance.", "text/plain")},
            data={"access_level": "restricted"},
        )
        assert uploaded.status_code == 200
        document_id = uploaded.json()["document"]["document_id"]

        denied = client.post(
            "/api/v1/chat/stream",
            json={"query": "SUP-STREAM-001", "access_levels": ["public"]},
        )
        assert denied.status_code == 200
        assert document_id not in denied.text

        allowed = client.post(
            "/api/v1/chat/stream",
            json={"query": "SUP-STREAM-001", "access_levels": ["restricted"]},
        )
        assert allowed.status_code == 200
        assert document_id in allowed.text
