import re
from typing import Any


class QueryProcessor:
    def rewrite(self, query: str, history: list[dict[str, Any]]) -> str:
        cleaned = " ".join(query.strip().split())
        if not history or not self._is_follow_up(cleaned):
            return cleaned
        previous = next(
            (item.get("content", "") for item in reversed(history) if item.get("role") == "user"),
            "",
        )
        if not previous:
            return cleaned
        return f"{previous}；追问：{cleaned}"[:1000]

    @staticmethod
    def _is_follow_up(query: str) -> bool:
        return bool(re.match(r"^(那|那么|它|这个|这个问题|上述|前面|其|该|还有|具体|分别|为什么|怎么|如何|呢|那请问)", query)) or len(query) <= 8
