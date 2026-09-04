import hashlib
import logging
import math
from typing import Any

from app.document.models import Chunk
from app.retrieval.models import RetrievalResult
from app.retrieval.tokenizer import tokenize

logger = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(self, model_name: str = "", dimension: int = 1024) -> None:
        self.model_name = model_name.strip()
        self.dimension = dimension
        self.model: Any | None = None
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []
        self.provider = "hash"
        self._vector_cache: dict[str, tuple[str, list[float]]] = {}
        self._model_attempted = False

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self._load_model()
        next_cache: dict[str, tuple[str, list[float]]] = {}
        missing: list[Chunk] = []
        for chunk in self.chunks:
            fingerprint = self._fingerprint(chunk.content)
            cached = self._vector_cache.get(chunk.chunk_id)
            if cached is not None and cached[0] == fingerprint:
                next_cache[chunk.chunk_id] = cached
            else:
                missing.append(chunk)
        if missing:
            encoded = self._encode([chunk.content for chunk in missing])
            for chunk, vector in zip(missing, encoded):
                next_cache[chunk.chunk_id] = (self._fingerprint(chunk.content), vector)
        self._vector_cache = next_cache
        self.vectors = [self._vector_cache[chunk.chunk_id][1] for chunk in self.chunks]

    def encode(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def search(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        if not self.chunks:
            return []
        query_vector = self._encode([query])[0]
        scored = [
            (self._cosine(query_vector, vector), chunk)
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [RetrievalResult(chunk=chunk, dense_score=score) for score, chunk in scored[:top_k]]

    def _load_model(self) -> None:
        if self._model_attempted:
            return
        self._model_attempted = True
        if not self.model_name:
            self.provider = "hash"
            self.model = None
            return
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            self.provider = "bge-m3" if "bge" in self.model_name.casefold() else "sentence-transformers"
        except Exception as exc:
            logger.warning("Embedding model unavailable, using deterministic fallback: %s", exc)
            self.model = None
            self.provider = "hash"

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if self.model is not None:
            try:
                encoded = self.model.encode(texts, normalize_embeddings=True)
                vectors = [self._normalize([float(value) for value in row]) for row in encoded]
                if vectors:
                    self.dimension = len(vectors[0])
                return vectors
            except Exception as exc:
                logger.warning("Embedding failed, using deterministic fallback: %s", exc)
                self.model = None
                self.provider = "hash"
        return [self._hash_encode(text) for text in texts]

    def _hash_encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = tokenize(text)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
            if len(token) > 1:
                for gram_start in range(len(token) - 1):
                    gram = token[gram_start : gram_start + 2]
                    gram_digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=16).digest()
                    gram_bucket = int.from_bytes(gram_digest[:4], "big") % self.dimension
                    vector[gram_bucket] += 0.25 if gram_digest[4] & 1 else -0.25
        return self._normalize(vector)

    @staticmethod
    def _fingerprint(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right))
