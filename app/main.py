from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.db.mysql import Database
from app.db.redis import Cache
from app.kg.neo4j_client import Neo4jClient
from app.kg.repository import KnowledgeRepository
from app.services.entity_service import EntityService
from app.services.intent_service import IntentService
from app.services.llm_service import LLMService
from app.services.qa_service import QAService


def create_app(settings: Settings | None = None) -> FastAPI:
    configure_logging()
    runtime_settings = settings or get_settings()
    project_root = Path(__file__).resolve().parents[1]
    database = Database(runtime_settings)
    cache = Cache(runtime_settings)
    neo4j_client = Neo4jClient(runtime_settings)
    knowledge_store_path = Path(runtime_settings.knowledge_store_path)
    if not knowledge_store_path.is_absolute():
        knowledge_store_path = project_root / knowledge_store_path
    repository = KnowledgeRepository(
        neo4j_client,
        project_root / "data" / "seed_knowledge.json",
        storage_path=knowledge_store_path,
    )
    entity_service = EntityService(repository)
    intent_service = IntentService()
    llm_service = LLMService(runtime_settings)
    qa_service = QAService(
        runtime_settings,
        repository,
        database,
        cache,
        entity_service,
        intent_service,
        llm_service,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        database.initialize()
        cache.initialize()
        repository.initialize()
        app.state.ready = True
        yield
        cache.close()
        database.close()
        neo4j_client.close()

    app = FastAPI(
        title=runtime_settings.app_name,
        description="基于知识图谱的可解释企业知识库问答服务",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=runtime_settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = runtime_settings
    app.state.database = database
    app.state.cache = cache
    app.state.neo4j_client = neo4j_client
    app.state.repository = repository
    app.state.entity_service = entity_service
    app.state.intent_service = intent_service
    app.state.llm_service = llm_service
    app.state.qa_service = qa_service
    app.state.ready = False
    app.include_router(router, prefix=runtime_settings.api_prefix)

    @app.get("/health", tags=["system"])
    @app.get(f"{runtime_settings.api_prefix}/health", tags=["system"])
    def health() -> dict:
        component_status = {
            "api": "ok",
            "mysql": "ok" if database.backend == "mysql" else database.backend,
            "redis": "ok" if cache.available else "memory-fallback",
            "neo4j": "ok" if neo4j_client.available else "memory-fallback",
            "llm": llm_service.status,
            "ner": "dictionary-matcher",
        }
        degraded_components = {"mysql", "neo4j"} - {
            key for key, value in component_status.items() if value in {"ok", "sqlite", "redis", "neo4j"}
        }
        return {
            "status": "degraded" if degraded_components else "ok",
            "ready": app.state.ready,
            "components": component_status,
        }

    static_dir = project_root / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
    return app


app = create_app()
