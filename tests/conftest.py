import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        fallback_database_path=str(tmp_path / "fallback.db"),
        redis_enabled=False,
        neo4j_enabled=False,
        llm_base_url="",
        admin_token="",
        knowledge_store_path=str(tmp_path / "knowledge_store.json"),
    )
    test_app = create_app(settings)
    with TestClient(test_app) as test_client:
        yield test_client
