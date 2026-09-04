import hashlib
import math
from collections import Counter

from app.document.models import Chunk
from app.retrieval.models import RetrievalResult
from app.retrieval.tokenizer import tokenize


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.term_frequencies: list[Counter[str]] = []
        self.document_frequencies: Counter[str] = Counter()
        self.average_length = 0.0

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self.term_frequencies = []
        self.document_frequencies = Counter()
        lengths = []
        for chunk in self.chunks:
            frequency = Counter(tokenize(chunk.content))
            self.term_frequencies.append(frequency)
            lengths.append(sum(frequency.values()))
            self.document_frequencies.update(frequency.keys())
        self.average_length = sum(lengths) / len(lengths) if lengths else 0.0

    def search(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks:
            return []
        query_set = set(query_tokens)
        document_count = len(self.chunks)
        scored: list[tuple[float, Chunk]] = []
        for index, (chunk, frequency) in enumerate(zip(self.chunks, self.term_frequencies)):
            document_length = sum(frequency.values())
            score = 0.0
            for token in query_set:
                term_frequency = frequency.get(token, 0)
                if not term_frequency:
                    continue
                inverse_frequency = math.log(1 + (document_count - self.document_frequencies[token] + 0.5) / (self.document_frequencies[token] + 0.5))
                denominator = term_frequency + self.k1 * (1 - self.b + self.b * document_length / max(self.average_length, 1))
                score += inverse_frequency * (term_frequency * (self.k1 + 1)) / denominator
            if query.strip() and query.casefold() in chunk.content.casefold():
                score += 2.0
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [RetrievalResult(chunk=chunk, bm25_score=score) for score, chunk in scored[:top_k]]

    def sparse_vector(self, text: str) -> dict[int, float]:
        frequency = Counter(tokenize(text))
        if not frequency:
            return {}
        document_count = len(self.chunks)
        document_length = sum(frequency.values())
        vector: dict[int, float] = {}
        for token, term_frequency in frequency.items():
            inverse_frequency = self._inverse_frequency(token, document_count)
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * document_length / max(self.average_length, 1)
            )
            weight = inverse_frequency * (term_frequency * (self.k1 + 1)) / max(denominator, 1e-12)
            if weight > 0:
                vector[self._token_id(token)] = float(weight)
        return vector

    def query_sparse_vector(self, text: str) -> dict[int, float]:
        frequency = Counter(tokenize(text))
        if not frequency:
            return {}
        document_count = len(self.chunks)
        return {
            self._token_id(token): float(self._inverse_frequency(token, document_count) * count)
            for token, count in frequency.items()
            if self._inverse_frequency(token, document_count) > 0
        }

    def _inverse_frequency(self, token: str, document_count: int) -> float:
        if document_count <= 0:
            return 0.0
        document_frequency = self.document_frequencies.get(token, 0)
        return math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))

    @staticmethod
    def _token_id(token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % 2_000_000_000 + 1
