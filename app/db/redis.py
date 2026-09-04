import json
import logging
import threading
import time
from typing import Any

import redis

from app.core.config import Settings

logger = logging.getLogger(__name__)


class Cache:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client: redis.Redis | None = None
        self.available = False
        self.backend = "memory"
        self._memory: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()

    def initialize(self) -> bool:
        if not self.settings.redis_enabled:
            return False
        try:
            self.client = redis.Redis(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                db=self.settings.redis_db,
                password=self.settings.redis_password or None,
                decode_responses=True,
                protocol=2,
                socket_connect_timeout=1.5,
                socket_timeout=1.5,
            )
            self.client.ping()
            self.available = True
            self.backend = "redis"
            return True
        except Exception as exc:
            logger.warning("Redis unavailable, using memory cache: %s", exc)
            self.client = None
            self.available = False
            self.backend = "memory"
            return False

    def get_json(self, key: str) -> Any | None:
        try:
            value = self.client.get(key) if self.client is not None else self._get_memory(key)
            return json.loads(value) if value else None
        except Exception as exc:
            logger.debug("Cache read failed: %s", exc)
            return None

    def set_json(self, key: str, value: Any, ttl: int | None = None) -> None:
        payload = json.dumps(value, ensure_ascii=False, default=str)
        expiry = ttl or self.settings.cache_ttl_seconds
        try:
            if self.client is not None:
                self.client.setex(key, expiry, payload)
            else:
                with self._lock:
                    self._memory[key] = (time.time() + expiry, payload)
        except Exception as exc:
            logger.debug("Cache write failed: %s", exc)

    def delete(self, key: str) -> None:
        try:
            if self.client is not None:
                self.client.delete(key)
            else:
                with self._lock:
                    self._memory.pop(key, None)
        except Exception as exc:
            logger.debug("Cache delete failed: %s", exc)

    def clear_prefix(self, prefix: str) -> None:
        try:
            if self.client is not None:
                keys = list(self.client.scan_iter(match=f"{prefix}*"))
                if keys:
                    self.client.delete(*keys)
            else:
                with self._lock:
                    for key in list(self._memory):
                        if key.startswith(prefix):
                            self._memory.pop(key, None)
        except Exception as exc:
            logger.debug("Cache namespace clear failed: %s", exc)

    def _get_memory(self, key: str) -> str | None:
        with self._lock:
            entry = self._memory.get(key)
            if entry is None:
                return None
            expires_at, payload = entry
            if expires_at < time.time():
                self._memory.pop(key, None)
                return None
            return payload

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
