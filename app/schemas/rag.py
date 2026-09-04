from typing import Any

from pydantic import BaseModel, Field


class Citation(BaseModel):
    document_id: str
    document: str
    filename: str
    version: str = "v1"
    page: int | None = None
    chapter: str = ""
    section: str = ""
    access_level: str = "internal"
    score: float = 0.0
    source: str = "document"


class RetrievalSummary(BaseModel):
    dense: int = 0
    bm25: int = 0
    fusion: int = 0
    reranked: int = 0
    strategy: str = "rrf"
    dense_provider: str = "hash"
    reranker_provider: str = "lexical"


class RetrievalSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=50)
    strategy: str | None = Field(default=None, pattern="^(rrf|weighted)$")
    access_levels: list[str] = Field(default_factory=list)
    rerank: bool = True


class DocumentUploadMetadata(BaseModel):
    department: str = Field(default="", max_length=100)
    access_level: str = Field(default="internal", max_length=40)
    version: str | None = Field(default=None, max_length=40)


def document_to_response(record: Any) -> dict[str, Any]:
    return record.to_dict() if hasattr(record, "to_dict") else dict(record)
