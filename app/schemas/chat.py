from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Entity(BaseModel):
    text: str
    type: str
    start: int
    end: int
    score: float = Field(ge=0, le=1)
    canonical_name: str | None = None


class IntentInfo(BaseModel):
    name: str
    label: str
    confidence: float = Field(ge=0, le=1)


class Evidence(BaseModel):
    entity: str
    relation: str
    value: str
    source: str = "medical_kg"
    description: str | None = None


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    session_id: str | None = Field(default=None, max_length=64)


class ChatResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    answer: str
    entities: list[Entity]
    intent: IntentInfo
    evidence: list[Evidence]
    grounded: bool
    latency_ms: float
    session_id: str
    message_id: int | None = None
    model: str = "grounded-fallback"
    cached: bool = False


class HistoryMessage(BaseModel):
    id: int
    role: str
    content: str
    intent: str | None = None
    entities: list[dict] = []
    evidence: list[dict] = []
    grounded: bool = False
    latency_ms: float | None = None
    created_at: datetime | str | None = None


class FeedbackRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    message_id: int | None = None
    rating: int = Field(ge=1, le=5)
    comment: str | None = Field(default=None, max_length=500)


class KnowledgeEntityInput(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    type: str = Field(min_length=1, max_length=50)
    description: str = ""
    aliases: list[str] = []


class KnowledgeRelationInput(BaseModel):
    source: str = Field(min_length=1, max_length=100)
    source_type: str = Field(min_length=1, max_length=50)
    relation: str = Field(min_length=1, max_length=60)
    target: str = Field(min_length=1, max_length=100)
    target_type: str = Field(min_length=1, max_length=50)


class KnowledgeImportRequest(BaseModel):
    entities: list[KnowledgeEntityInput] = Field(default_factory=list)
    relations: list[KnowledgeRelationInput] = Field(default_factory=list)

