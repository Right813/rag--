from functools import lru_cache
from urllib.parse import quote_plus

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "知库智答"
    app_env: str = "development"
    api_prefix: str = "/api/v1"

    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "root"
    mysql_database: str = "rag_kg"
    database_url: str | None = None

    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = ""
    redis_enabled: bool = True

    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "knowledge"
    neo4j_enabled: bool = True

    llm_base_url: str | None = ""
    llm_api_key: str | None = ""
    llm_model: str = "qwen2.5:7b"
    llm_timeout_seconds: float = 30.0

    cache_ttl_seconds: int = 300
    admin_token: str | None = ""
    cors_origins: str = "*"
    max_query_length: int = 500
    fallback_database_path: str = "data/rag_kg_fallback.db"

    document_storage_path: str = "data/documents.json"
    document_upload_dir: str = "data/uploads"
    max_upload_size_bytes: int = 20 * 1024 * 1024
    chunk_size: int = 600
    chunk_overlap: int = 80
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    reranker_model: str = ""
    dense_top_k: int = 20
    bm25_top_k: int = 20
    retrieval_top_k: int = 5
    fusion_strategy: str = "rrf"
    fusion_alpha: float = Field(default=0.7, ge=0, le=1)
    max_context_tokens: int = 6000
    no_answer_threshold: float = Field(default=0.14, ge=0, le=1)
    ocr_enabled: bool = True
    milvus_enabled: bool = True
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: str = ""
    milvus_db_name: str = "default"
    milvus_collection: str = "medical_knowledge_chunks"
    milvus_timeout_seconds: float = 10.0
    knowledge_store_path: str = "data/knowledge_store.json"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return self.database_url
        user = quote_plus(self.mysql_user)
        password = quote_plus(self.mysql_password)
        return f"mysql+pymysql://{user}:{password}@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"

    @property
    def llm_configured(self) -> bool:
        if not self.llm_base_url:
            return False
        return bool(self.llm_api_key or "localhost" in self.llm_base_url or "127.0.0.1" in self.llm_base_url)

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
