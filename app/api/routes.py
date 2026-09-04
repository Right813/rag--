import csv
import io
import json
import logging
from pathlib import Path

from fastapi import APIRouter, File, Header, HTTPException, Query, Request, UploadFile, status

from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    KnowledgeImportRequest,
)

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
