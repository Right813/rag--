from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ParsedPage:
    content: str
    page: int | None = None
    chapter: str = ""
    section: str = ""


@dataclass
class DocumentRecord:
    document_id: str
    filename: str
    version: str
    file_type: str
    size_bytes: int
    raw_path: str
    department: str = ""
    access_level: str = "internal"
    status: str = "processing"
    chunk_count: int = 0
    error: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    content: str
    filename: str
    version: str
    file_type: str
    page: int | None = None
    chapter: str = ""
    section: str = ""
    department: str = ""
    access_level: str = "internal"
    source: str = "document"
    entity: str = ""
    relation: str = ""
    value: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
