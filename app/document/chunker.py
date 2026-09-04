import re
import uuid

from app.document.cleaner import clean_pages
from app.document.models import Chunk, DocumentRecord, ParsedPage
from app.document.parsers import detect_structure


def estimate_tokens(text: str) -> int:
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z0-9_]+", text))
    return chinese + latin + max(0, len(text) - chinese - sum(len(item) for item in re.findall(r"[A-Za-z0-9_]+", text))) // 4


class StructureAwareChunker:
    def __init__(self, chunk_size: int = 600, overlap: int = 80) -> None:
        self.chunk_size = max(100, chunk_size)
        self.overlap = max(0, min(overlap, self.chunk_size // 2))

    def split(self, pages: list[ParsedPage], document: DocumentRecord) -> list[Chunk]:
        current_chapter = ""
        current_section = ""
        chunks: list[Chunk] = []
        for page in clean_pages(pages):
            paragraphs = [part.strip() for part in re.split(r"\n\s*\n|\n", page.content) if part.strip()]
            current_text: list[str] = []
            current_tokens = 0
            chapter = page.chapter or current_chapter
            section = page.section or current_section
            for paragraph in paragraphs:
                detected_chapter, detected_section = detect_structure(paragraph, chapter, section)
                if detected_chapter != chapter and detected_chapter == paragraph:
                    if current_text:
                        chunks.extend(self._make_chunks("\n".join(current_text), document, page.page, chapter, section))
                        current_text, current_tokens = [], 0
                    chapter, section = detected_chapter, ""
                    continue
                if detected_section != section and detected_section == paragraph:
                    if current_text:
                        chunks.extend(self._make_chunks("\n".join(current_text), document, page.page, chapter, section))
                        current_text, current_tokens = [], 0
                    section = detected_section
                    continue
                paragraph_tokens = estimate_tokens(paragraph)
                if current_text and current_tokens + paragraph_tokens > self.chunk_size:
                    chunks.extend(self._make_chunks("\n".join(current_text), document, page.page, chapter, section))
                    overlap_text = self._overlap_text(current_text)
                    current_text = [overlap_text] if overlap_text else []
                    current_tokens = estimate_tokens(overlap_text)
                if paragraph_tokens > self.chunk_size:
                    if current_text:
                        chunks.extend(self._make_chunks("\n".join(current_text), document, page.page, chapter, section))
                        current_text, current_tokens = [], 0
                    long_chunks = self._split_long(paragraph)
                    chunks.extend(self._make_chunks(item, document, page.page, chapter, section) for item in long_chunks)
                else:
                    current_text.append(paragraph)
                    current_tokens += paragraph_tokens
            if current_text:
                chunks.extend(self._make_chunks("\n".join(current_text), document, page.page, chapter, section))
            current_chapter, current_section = chapter, section
        return [chunk for group in chunks for chunk in (group if isinstance(group, list) else [group]) if chunk.content]

    def _make_chunks(self, text: str, document: DocumentRecord, page: int | None, chapter: str, section: str) -> list[Chunk]:
        return [
            Chunk(
                chunk_id=uuid.uuid4().hex,
                document_id=document.document_id,
                content=text.strip(),
                filename=document.filename,
                version=document.version,
                file_type=document.file_type,
                page=page,
                chapter=chapter,
                section=section,
                department=document.department,
                access_level=document.access_level,
            )
        ]

    def _split_long(self, text: str) -> list[str]:
        sentences = [item.strip() for item in re.split(r"(?<=[。！？；.!?;])\s*", text) if item.strip()]
        if not sentences:
            sentences = [text]
        result: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for sentence in sentences:
            sentence_tokens = estimate_tokens(sentence)
            if current and current_tokens + sentence_tokens > self.chunk_size:
                result.append("".join(current))
                overlap = self._overlap_text(current)
                current = [overlap] if overlap else []
                current_tokens = estimate_tokens(overlap)
            if sentence_tokens > self.chunk_size:
                for start in range(0, len(sentence), max(1, self.chunk_size - self.overlap)):
                    result.append(sentence[start : start + self.chunk_size])
                current, current_tokens = [], 0
            else:
                current.append(sentence)
                current_tokens += sentence_tokens
        if current:
            result.append("".join(current))
        return result

    def _overlap_text(self, paragraphs: list[str]) -> str:
        if not paragraphs or self.overlap == 0:
            return ""
        text = paragraphs[-1]
        return text[-self.overlap :]
