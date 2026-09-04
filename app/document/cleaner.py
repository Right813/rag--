import re
import unicodedata

from app.document.models import ParsedPage


def clean_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\u0000", " ").replace("\r\n", "\n").replace("\r", "\n")
    normalized = "".join(char if char in {"\n", "\t"} or not unicodedata.category(char).startswith("C") else " " for char in normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"(?m)^\s*(?:第\s*\d+\s*页|Page\s+\d+)\s*$", "", normalized, flags=re.IGNORECASE)
    return "\n".join(line.strip() for line in normalized.splitlines()).strip()


def clean_pages(pages: list[ParsedPage]) -> list[ParsedPage]:
    cleaned = [
        ParsedPage(content=clean_text(page.content), page=page.page, chapter=page.chapter, section=page.section)
        for page in pages
    ]
    non_empty = [page for page in cleaned if page.content]
    repeated_lines: dict[str, int] = {}
    for page in non_empty:
        seen = set(page.content.splitlines())
        for line in seen:
            if len(line) >= 4:
                repeated_lines[line] = repeated_lines.get(line, 0) + 1
    threshold = max(2, len(non_empty) // 2) if non_empty else 2
    headers = {line for line, count in repeated_lines.items() if count >= threshold and len(line) < 100}
    if not headers:
        return non_empty
    result = []
    for page in non_empty:
        content = "\n".join(line for line in page.content.splitlines() if line not in headers).strip()
        if content:
            result.append(ParsedPage(content=content, page=page.page, chapter=page.chapter, section=page.section))
    return result
