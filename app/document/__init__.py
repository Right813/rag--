from app.document.models import Chunk, DocumentRecord, ParsedPage

__all__ = ["Chunk", "DocumentRecord", "ParsedPage", "DocumentService"]


def __getattr__(name: str):
    if name == "DocumentService":
        from app.document.service import DocumentService

        return DocumentService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
