from typing import Any

from app.retrieval.models import RetrievalResult


def build_citations(results: list[RetrievalResult]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None, str, str]] = set()
    for result in results:
        chunk = result.chunk
        key = (chunk.document_id, chunk.page, chunk.chapter, chunk.section)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "document_id": chunk.document_id,
                "document": chunk.filename,
                "filename": chunk.filename,
                "version": chunk.version,
                "category": chunk.category,
                "page": chunk.page,
                "chapter": chunk.chapter,
                "section": chunk.section,
                "access_level": chunk.access_level,
                "score": round(result.score, 6),
                "source": chunk.source,
            }
        )
    return citations
