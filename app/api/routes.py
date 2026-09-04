import csv
import io
import json
import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, Header, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    KnowledgeImportRequest,
)
from app.schemas.rag import RetrievalSearchRequest

logger = logging.getLogger(__name__)
router = APIRouter()


def _services(request: Request):
    return request.app.state


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    try:
        return _services(request).qa_service.chat(payload.query, payload.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat request failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="问答服务暂时不可用，请稍后重试。",
        ) from exc


@router.get("/history")
def history(request: Request, session_id: str = Query(min_length=1, max_length=64)) -> dict:
    return {"session_id": session_id, "messages": _services(request).qa_service.history(session_id)}


@router.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
def feedback(payload: FeedbackRequest, request: Request) -> None:
    _services(request).database.save_feedback(
        payload.session_id,
        payload.message_id,
        payload.rating,
        payload.comment,
    )


@router.get("/knowledge/stats")
def knowledge_stats(request: Request) -> dict:
    app_state = _services(request)
    return {
        "graph": app_state.repository.stats(),
        "metrics": app_state.qa_service.metrics(),
        "backends": {
            "database": app_state.database.backend,
            "cache": app_state.cache.backend,
            "knowledge_graph": "neo4j" if app_state.neo4j_client.available else "memory",
            "llm": app_state.llm_service.status,
        },
    }


@router.post("/reload")
def reload_knowledge(request: Request, x_admin_token: str | None = Header(default=None)) -> dict:
    app_state = _services(request)
    _require_admin(app_state.settings.admin_token, x_admin_token)
    try:
        app_state.repository.initialize()
        app_state.cache.clear_prefix("qa:v1:")
        app_state.cache.clear_prefix("qa:doc:v1:")
        document_service = getattr(app_state.qa_service, "document_service", None)
        if document_service is not None:
            document_service.refresh_legacy()
        return {"status": "ok", "graph": app_state.repository.stats()}
    except Exception as exc:
        logger.exception("Knowledge reload failed: %s", exc)
        raise HTTPException(status_code=500, detail="知识库重载失败") from exc


@router.post("/knowledge/import")
def import_knowledge(
    payload: KnowledgeImportRequest,
    request: Request,
    x_admin_token: str | None = Header(default=None),
) -> dict:
    app_state = _services(request)
    _require_admin(app_state.settings.admin_token, x_admin_token)
    try:
        result = app_state.repository.merge(
            [entity.model_dump() for entity in payload.entities],
            [relation.model_dump() for relation in payload.relations],
        )
        app_state.cache.clear_prefix("qa:v1:")
        app_state.cache.clear_prefix("qa:doc:v1:")
        document_service = getattr(app_state.qa_service, "document_service", None)
        if document_service is not None:
            document_service.refresh_legacy()
        return {"status": "ok", "graph": result}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Knowledge import failed: %s", exc)
        raise HTTPException(status_code=500, detail="知识导入失败") from exc


@router.post("/knowledge/import-file")
async def import_knowledge_file(
    request: Request,
    file: UploadFile = File(...),
    x_admin_token: str | None = Header(default=None),
) -> dict:
    app_state = _services(request)
    _require_admin(app_state.settings.admin_token, x_admin_token)
    filename = file.filename or ""
    suffix = Path(filename).suffix.casefold()
    if suffix not in {".json", ".csv"}:
        raise HTTPException(status_code=415, detail="仅支持 JSON 或 CSV 文件")
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="导入文件不能超过 5MB")
    try:
        payload = _parse_import_file(content, suffix)
        validated = KnowledgeImportRequest.model_validate(payload)
        result = app_state.repository.merge(
            [entity.model_dump() for entity in validated.entities],
            [relation.model_dump() for relation in validated.relations],
        )
        app_state.cache.clear_prefix("qa:v1:")
        app_state.cache.clear_prefix("qa:doc:v1:")
        document_service = getattr(app_state.qa_service, "document_service", None)
        if document_service is not None:
            document_service.refresh_legacy()
        return {"status": "ok", "filename": filename, "graph": result}
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"文件内容无效：{exc}") from exc
    except Exception as exc:
        logger.exception("Knowledge file import failed: %s", exc)
        raise HTTPException(status_code=500, detail="知识文件导入失败") from exc


@router.get("/suggestions")
def suggestions() -> dict:
    return {
        "items": [
            {"title": "高血压症状", "query": "高血压有哪些常见症状？", "tag": "疾病 · 症状"},
            {"title": "糖尿病检查", "query": "糖尿病需要做哪些检查？", "tag": "疾病 · 检查"},
            {"title": "感冒用药", "query": "感冒可以使用哪些药物？", "tag": "疾病 · 用药"},
            {"title": "氨氯地平作用", "query": "氨氯地平有什么作用？", "tag": "药物 · 作用"},
        ]
    }


def _require_admin(configured_token: str | None, provided_token: str | None) -> None:
    if configured_token and configured_token != provided_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="管理员凭证无效")


def _parse_import_file(content: bytes, suffix: str) -> dict:
    if suffix == ".json":
        payload = json.loads(content.decode("utf-8"))
        if isinstance(payload, list):
            if payload and {"source", "target"}.issubset(payload[0]):
                return {"entities": [], "relations": payload}
            return {"entities": payload, "relations": []}
        if not isinstance(payload, dict):
            raise ValueError("JSON 顶层必须是对象或数组")
        return payload

    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    if not rows:
        raise ValueError("CSV 文件没有数据")
    headers = set(rows[0])
    if {"source", "source_type", "relation", "target", "target_type"}.issubset(headers):
        return {
            "entities": [],
            "relations": [
                {
                    "source": row.get("source", "").strip(),
                    "source_type": row.get("source_type", "").strip(),
                    "relation": row.get("relation", "").strip(),
                    "target": row.get("target", "").strip(),
                    "target_type": row.get("target_type", "").strip(),
                }
                for row in rows
            ],
        }
    if {"name", "type"}.issubset(headers):
        return {
            "entities": [
                {
                    "name": row.get("name", "").strip(),
                    "type": row.get("type", "").strip(),
                    "description": row.get("description", "").strip(),
                    "aliases": [
                        alias.strip()
                        for alias in row.get("aliases", "").replace("|", ",").split(",")
                        if alias.strip()
                    ],
                }
                for row in rows
            ],
            "relations": [],
        }
    raise ValueError("CSV 表头需包含实体列 name,type 或关系列 source,source_type,relation,target,target_type")


def _documents(request: Request):
    from app.services.document_qa import get_document_service

    return get_document_service(_services(request).qa_service)


def _clear_document_cache(request: Request) -> None:
    _services(request).cache.clear_prefix("qa:doc:v1:")


@router.post("/chat/stream")
def chat_stream(payload: ChatRequest, request: Request):
    try:
        response = _services(request).qa_service.chat(payload.query, payload.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    def events():
        yield f"event: meta\ndata: {json.dumps({'session_id': response.session_id, 'message_id': response.message_id}, ensure_ascii=False)}\n\n"
        answer = response.answer
        for start in range(0, len(answer), 48):
            yield f"event: token\ndata: {json.dumps({'text': answer[start:start + 48]}, ensure_ascii=False)}\n\n"
        yield f"event: citations\ndata: {json.dumps({'citations': [item.model_dump() for item in response.citations]}, ensure_ascii=False, default=str)}\n\n"
        yield f"event: done\ndata: {json.dumps(response.model_dump(mode='json'), ensure_ascii=False, default=str)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/documents/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    department: str = Form(default=""),
    access_level: str = Form(default="internal"),
    version: str | None = Form(default=None),
    x_admin_token: str | None = Header(default=None),
) -> dict:
    app_state = _services(request)
    _require_admin(app_state.settings.admin_token, x_admin_token)
    try:
        content = await file.read()
        record = _documents(request).upload(
            file.filename or "",
            content,
            department=department,
            access_level=access_level,
            version=version,
        )
        _clear_document_cache(request)
        return {"status": "ok", "document": record.to_dict()}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document upload failed: %s", exc)
        raise HTTPException(status_code=500, detail="文档处理失败，请检查文件格式") from exc


@router.get("/documents/stats")
def document_stats(request: Request) -> dict:
    return _documents(request).stats()


@router.get("/documents")
def list_documents(
    request: Request,
    status_filter: str | None = Query(default=None, alias="status", max_length=20),
    search: str | None = Query(default=None, max_length=100),
) -> dict:
    records = _documents(request).list(status=status_filter, search=search)
    return {"items": [record.to_dict() for record in records], "total": len(records)}


@router.get("/documents/{document_id}")
def document_detail(document_id: str, request: Request) -> dict:
    service = _documents(request)
    try:
        record = service.get(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    chunks = [
        {
            "chunk_id": chunk.chunk_id,
            "page": chunk.page,
            "chapter": chunk.chapter,
            "section": chunk.section,
            "preview": chunk.content[:280],
        }
        for chunk in service.chunks.get(document_id, [])
    ]
    return {"document": record.to_dict(), "chunks": chunks}


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str, request: Request, x_admin_token: str | None = Header(default=None)) -> None:
    app_state = _services(request)
    _require_admin(app_state.settings.admin_token, x_admin_token)
    try:
        _documents(request).delete(document_id)
        _clear_document_cache(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/documents/{document_id}/reindex")
def reindex_document(document_id: str, request: Request, x_admin_token: str | None = Header(default=None)) -> dict:
    app_state = _services(request)
    _require_admin(app_state.settings.admin_token, x_admin_token)
    try:
        record = _documents(request).reindex(document_id)
        _clear_document_cache(request)
        return {"status": "ok", "document": record.to_dict()}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/retrieval/search")
def retrieval_search(payload: RetrievalSearchRequest, request: Request) -> dict:
    data = payload
    results = _documents(request).search(
        data.query,
        top_k=data.top_k,
        strategy=data.strategy,
        access_levels=set(data.access_levels),
        rerank=data.rerank,
    )
    service = _documents(request)
    return {
        "query": data.query,
        "results": [item.to_dict() for item in results],
        "total": len(results),
        "retrieval": {
            "fusion": len(results),
            "reranked": len(results) if data.rerank else 0,
            "strategy": data.strategy or getattr(service.settings, "fusion_strategy", "rrf"),
            "dense_provider": service.retriever.dense_provider,
            "reranker_provider": service.retriever.reranker_provider,
        },
    }
