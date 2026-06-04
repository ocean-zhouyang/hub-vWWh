"""LLM Semantic Cache — cache prompt→response pairs with vector similarity lookup.

When a new prompt is "close enough" to a cached one, return the cached response
instead of calling the LLM again. Saves cost and latency.
"""

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from redis import Redis

from redisvl.client import RedisVL
from redisvl.query import VectorRangeQuery
from redisvl.schema import IndexSchema


def _hash_prompt(prompt: str) -> str:
    """Short deterministic hash for cache key generation."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


@dataclass
class CacheHit:
    """A matched cache entry from semantic search."""
    entry_id: str
    prompt: str
    response: str
    vector_distance: float
    inserted_at: float | None = None
    metadata: dict[str, Any] | None = None


class SemanticCache:
    """Cache LLM responses keyed by semantic similarity of prompts.

    Usage:
        cache = SemanticCache("my-llm", vector_dims=768, distance_threshold=0.15)
        cache.store("What is Redis?", "Redis is an in-memory DB.", vector=[0.1, ...])

        hit = cache.check("Tell me about Redis", vector=[0.11, ...])
        if hit:
            print(f"Cache hit! {hit.response}")  # avoids calling LLM
        else:
            response = call_llm(...)
            cache.store("Tell me about Redis", response, vector=[0.11, ...])
    """

    _SCHEMA_FIELDS = [
        {"name": "prompt", "type": "text"},
        {"name": "response", "type": "text"},
        {"name": "prompt_vector", "type": "vector", "algorithm": "flat",
         "dims": 768, "distance_metric": "cosine"},
        {"name": "inserted_at", "type": "numeric"},
    ]

    def __init__(
        self,
        name: str = "llmcache",
        vector_dims: int = 768,
        distance_threshold: float = 0.15,
        ttl: int | None = None,
        redis_client: Redis | None = None,
        redis_url: str = "redis://localhost:6379",
        overwrite: bool = False,
        **redis_kwargs,
    ):
        self._name = name
        self._distance_threshold = distance_threshold
        self._ttl = ttl

        fields = [dict(f) for f in self._SCHEMA_FIELDS]
        for f in fields:
            if f["name"] == "prompt_vector":
                f["dims"] = vector_dims

        self._rvl = RedisVL(url=redis_url, **redis_kwargs) if redis_client is None else None

        if redis_client:
            self._rvl = RedisVL.__new__(RedisVL)
            self._rvl._client = redis_client
            self._rvl._url = ""
            self._rvl._redis_kwargs = {}
            self._rvl._schemas = {}

        self._schema = IndexSchema.from_dict({
            "index": {"name": name, "prefix": name, "storage_type": "hash"},
            "fields": fields,
        })
        self._rvl.create_index(name, self._schema, overwrite=overwrite)

    # ── core ───────────────────────────────────────────────────
    def store(
        self,
        prompt: str,
        response: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> str:
        """Store a prompt/response pair with its vector embedding.

        Returns the entry_id (hash of the prompt).
        """
        entry_id = _hash_prompt(prompt)
        doc = {
            "entry_id": entry_id,
            "prompt": prompt,
            "response": response,
            "prompt_vector": vector,
            "inserted_at": time.time(),
        }
        if metadata:
            doc["metadata"] = metadata

        keys = self._rvl.load([doc], self._name, id_field="entry_id")

        actual_ttl = ttl if ttl is not None else self._ttl
        if actual_ttl and keys:
            for key in keys:
                self._rvl.client.expire(key, actual_ttl)

        return entry_id

    def check(
        self,
        vector: list[float],
        top_k: int = 3,
        distance_threshold: float | None = None,
    ) -> list[CacheHit]:
        """Search for semantically similar cached prompts.

        Args:
            vector: The query prompt's embedding.
            top_k: Max cache hits to return.
            distance_threshold: Override the default threshold.

        Returns:
            List of CacheHit objects sorted by distance (closest first).
            Empty list means cache miss.
        """
        threshold = distance_threshold or self._distance_threshold
        q = VectorRangeQuery(
            vector=vector,
            vector_field_name="prompt_vector",
            distance_threshold=threshold,
            return_fields=["prompt", "response", "inserted_at", "entry_id"],
            distance_metric="COSINE",
        )
        results = self._rvl.search(q, self._name)

        hits = []
        for r in results[:top_k]:
            hits.append(CacheHit(
                entry_id=r.get("entry_id", ""),
                prompt=r.get("prompt", ""),
                response=r.get("response", ""),
                vector_distance=float(r.get("vector_distance", 1.0)),
                inserted_at=float(r.get("inserted_at", 0)) if r.get("inserted_at") else None,
                metadata=r.get("metadata"),
            ))

        # Refresh TTL on hits
        if self._ttl and hits:
            pipe = self._rvl.client.pipeline()
            for h in hits:
                key = f"{self._name}:{h.entry_id}"
                pipe.expire(key, self._ttl)
            pipe.execute()

        return hits

    # ── maintenance ────────────────────────────────────────────
    def set_threshold(self, threshold: float) -> None:
        """Update the default distance threshold."""
        if not 0 <= threshold <= 2:
            raise ValueError("Distance threshold must be in [0, 2] for cosine distance")
        self._distance_threshold = threshold

    def drop(self, entry_id: str) -> bool:
        """Delete a specific cache entry by ID."""
        key = f"{self._name}:{entry_id}"
        return self._rvl.client.delete(key) > 0

    def clear(self) -> int:
        """Remove all cached entries but keep the index."""
        return self._rvl.clear_index(self._name)

    def delete(self) -> None:
        """Delete the index and all cache entries."""
        self._rvl.drop_index(self._name)
        self._rvl.disconnect()

    def disconnect(self) -> None:
        self._rvl.disconnect()

    @property
    def threshold(self) -> float:
        return self._distance_threshold

    def __repr__(self) -> str:
        return f"SemanticCache(name='{self._name}', threshold={self._distance_threshold})"
