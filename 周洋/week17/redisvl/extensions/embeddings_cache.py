"""Embeddings cache — avoids redundant embedding API calls.

Uses deterministic hash keys (content + model_name) for exact lookup.
No RediSearch index needed — pure Redis HASH operations.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from redis import Redis


def _hash_key(content: str | bytes, model_name: str) -> str:
    """Generate a deterministic hash from content and model name."""
    if isinstance(content, bytes):
        content = content.hex()
    raw = f"{content}:{model_name}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


@dataclass
class CacheEntry:
    """A single cached embedding entry."""
    entry_id: str
    content: str
    model_name: str
    embedding: list[float]
    inserted_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] | None = None


class EmbeddingsCache:
    """Cache embeddings by content+model to avoid duplicate API calls.

    Usage:
        cache = EmbeddingsCache("embeds", ttl=3600, redis_url="redis://localhost:6379")
        cache.set("hello world", "text-embedding-3", [0.1, 0.2, ...])
        emb = cache.get("hello world", "text-embedding-3")  # -> list[float] or None
        exists = cache.exists("hello world", "text-embedding-3")
    """

    def __init__(
        self,
        name: str = "embedcache",
        ttl: int | None = None,
        redis_client: Redis | None = None,
        redis_url: str = "redis://localhost:6379",
        **redis_kwargs,
    ):
        self.name = name
        self.ttl = ttl
        self._client = redis_client
        self._url = redis_url
        self._redis_kwargs = redis_kwargs

    @property
    def client(self) -> Redis:
        if self._client is None:
            self._client = Redis.from_url(self._url, **self._redis_kwargs)
        return self._client

    def _make_key(self, entry_id: str) -> str:
        return f"{self.name}:{entry_id}"

    # ── single operations ──────────────────────────────────────
    def set(
        self,
        content: str | bytes,
        model_name: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> str:
        """Store an embedding. Returns the entry_id."""
        entry_id = _hash_key(content, model_name)
        key = self._make_key(entry_id)
        data = {
            "entry_id": entry_id,
            "content": content if isinstance(content, str) else content.hex(),
            "model_name": model_name,
            "embedding": json.dumps(embedding),
            "inserted_at": str(time.time()),
        }
        if metadata:
            data["metadata"] = json.dumps(metadata)

        pipe = self.client.pipeline()
        pipe.delete(key)
        pipe.hset(key, mapping=data)
        actual_ttl = ttl if ttl is not None else self.ttl
        if actual_ttl:
            pipe.expire(key, actual_ttl)
        pipe.execute()
        return entry_id

    def get(self, content: str | bytes, model_name: str) -> list[float] | None:
        """Retrieve a cached embedding, or None if not found."""
        entry_id = _hash_key(content, model_name)
        key = self._make_key(entry_id)
        raw = self.client.hget(key, "embedding")
        if raw is None:
            return None
        # Refresh TTL on hit
        if self.ttl:
            self.client.expire(key, self.ttl)
        return json.loads(raw)

    def exists(self, content: str | bytes, model_name: str) -> bool:
        """Check whether an embedding is cached."""
        entry_id = _hash_key(content, model_name)
        key = self._make_key(entry_id)
        return self.client.exists(key) > 0

    def drop(self, content: str | bytes, model_name: str) -> bool:
        """Delete a cached embedding. Returns True if it was deleted."""
        entry_id = _hash_key(content, model_name)
        key = self._make_key(entry_id)
        return self.client.delete(key) > 0

    # ── batch operations ───────────────────────────────────────
    def mset(
        self,
        items: list[tuple[str | bytes, str, list[float]]],
        ttl: int | None = None,
    ) -> list[str]:
        """Batch-store multiple embeddings. Returns list of entry_ids."""
        pipe = self.client.pipeline(transaction=False)
        entry_ids = []
        actual_ttl = ttl if ttl is not None else self.ttl

        for content, model_name, embedding in items:
            entry_id = _hash_key(content, model_name)
            key = self._make_key(entry_id)
            entry_ids.append(entry_id)
            data = {
                "entry_id": entry_id,
                "content": content if isinstance(content, str) else content.hex(),
                "model_name": model_name,
                "embedding": json.dumps(embedding),
                "inserted_at": str(time.time()),
            }
            pipe.delete(key)
            pipe.hset(key, mapping=data)
            if actual_ttl:
                pipe.expire(key, actual_ttl)

        pipe.execute()
        return entry_ids

    def mget(
        self, items: list[tuple[str | bytes, str]]
    ) -> list[list[float] | None]:
        """Batch-retrieve multiple cached embeddings."""
        pipe = self.client.pipeline(transaction=False)
        keys = []
        for content, model_name in items:
            entry_id = _hash_key(content, model_name)
            keys.append(self._make_key(entry_id))
        for key in keys:
            pipe.hget(key, "embedding")
        results = pipe.execute()
        out: list[list[float] | None] = []
        for r in results:
            out.append(json.loads(r) if r else None)
        return out

    # ── maintenance ────────────────────────────────────────────
    def clear(self) -> int:
        """Remove all cached entries. Returns count removed."""
        pattern = f"{self.name}:*"
        removed = 0
        cursor = 0
        while True:
            cursor, keys = self.client.scan(cursor=cursor, match=pattern, count=500)
            if keys:
                removed += self.client.delete(*keys)
            if cursor == 0:
                break
        return removed

    def disconnect(self) -> None:
        """Close the Redis connection."""
        if self._client:
            self._client.close()
            self._client = None

    def __repr__(self) -> str:
        return f"EmbeddingsCache(name='{self.name}', ttl={self.ttl})"
