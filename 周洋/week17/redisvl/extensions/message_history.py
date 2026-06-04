"""Message history — store and retrieve chat conversation history.

Two classes:
- MessageHistory: basic session-based retrieval by recency
- SemanticMessageHistory: adds semantic (vector) search over messages
"""

import time
import uuid
from dataclasses import dataclass
from typing import Any

from redis import Redis

from redisvl.client import RedisVL
from redisvl.query import FilterQuery, VectorQuery
from redisvl.schema import IndexSchema


def _make_session_tag() -> str:
    """Generate a unique session identifier."""
    return uuid.uuid4().hex[:12]


class MessageHistory:
    """Store and retrieve multi-turn chat messages.

    Usage:
        hist = MessageHistory("my-bot", redis_url="redis://localhost:6379")
        hist.store("What's Redis?", "Redis is an in-memory database.")
        recent = hist.get_recent(top_k=5)
        count = hist.count()
    """

    _SCHEMA_FIELDS = [
        {"name": "role", "type": "tag"},
        {"name": "content", "type": "text"},
        {"name": "session_tag", "type": "tag"},
        {"name": "timestamp", "type": "numeric"},
    ]

    def __init__(
        self,
        name: str,
        session_tag: str | None = None,
        redis_client: Redis | None = None,
        redis_url: str = "redis://localhost:6379",
        overwrite: bool = False,
        **redis_kwargs,
    ):
        self._name = name
        self._session_tag = session_tag or _make_session_tag()
        self._rvl = RedisVL(url=redis_url, **redis_kwargs) if redis_client is None else None

        if redis_client:
            self._rvl = RedisVL.__new__(RedisVL)
            self._rvl._client = redis_client
            self._rvl._url = ""
            self._rvl._redis_kwargs = {}
            self._rvl._schemas = {}

        self._schema = IndexSchema.from_dict({
            "index": {"name": name, "prefix": name, "storage_type": "hash"},
            "fields": list(self._SCHEMA_FIELDS),
        })
        self._rvl.create_index(name, self._schema, overwrite=overwrite)

    @property
    def session_tag(self) -> str:
        return self._session_tag

    # ── write ──────────────────────────────────────────────────
    def store(
        self,
        prompt: str,
        response: str,
        session_tag: str | None = None,
    ) -> list[str]:
        """Store a user/assistant turn. Returns the Redis keys."""
        tag = session_tag or self._session_tag
        ts = time.time()
        messages = [
            {"role": "user", "content": prompt, "session_tag": tag, "timestamp": ts},
            {"role": "assistant", "content": response, "session_tag": tag, "timestamp": ts + 0.001},
        ]
        return self._rvl.load(messages, self._name, id_field="role")

    def add_message(
        self,
        role: str,
        content: str,
        session_tag: str | None = None,
    ) -> list[str]:
        """Store a single message with the given role."""
        tag = session_tag or self._session_tag
        msg = {"role": role, "content": content, "session_tag": tag, "timestamp": time.time()}
        return self._rvl.load([msg], self._name, id_field="role")

    def add_messages(
        self,
        messages: list[dict[str, str]],
        session_tag: str | None = None,
    ) -> list[str]:
        """Batch-add messages. Each dict needs 'role' and 'content'."""
        tag = session_tag or self._session_tag
        ts = time.time()
        docs = []
        for i, m in enumerate(messages):
            docs.append({
                "role": m["role"],
                "content": m["content"],
                "session_tag": tag,
                "timestamp": ts + i * 0.001,
            })
        return self._rvl.load(docs, self._name, id_field="role")

    # ── read ───────────────────────────────────────────────────
    def get_recent(
        self,
        top_k: int = 10,
        session_tag: str | None = None,
        role: str | None = None,
    ) -> list[dict]:
        """Get most recent messages, optionally filtered by session/role."""
        parts = []
        tag = session_tag or self._session_tag
        if tag:
            parts.append(f"@session_tag:{{{tag}}}")
        if role:
            parts.append(f"@role:{{{role}}}")
        expr = " ".join(parts) if parts else "*"

        q = FilterQuery(
            filter_expr=expr,
            top_k=top_k,
            sort_by="timestamp DESC",
            return_fields=["role", "content", "session_tag", "timestamp"],
        )
        return self._rvl.search(q, self._name)

    def count(self, session_tag: str | None = None) -> int:
        """Count messages in a session."""
        tag = session_tag or self._session_tag
        expr = f"@session_tag:{{{tag}}}" if tag else "*"
        q = FilterQuery(filter_expr=expr, top_k=10000)
        results = self._rvl.search(q, self._name)
        return len(results)

    # ── maintenance ────────────────────────────────────────────
    def clear(self) -> int:
        """Remove all messages but keep the index."""
        return self._rvl.clear_index(self._name)

    def delete(self) -> None:
        """Delete the index and all messages."""
        self._rvl.drop_index(self._name)
        self._rvl.disconnect()

    def disconnect(self) -> None:
        self._rvl.disconnect()

    def __repr__(self) -> str:
        return f"MessageHistory(name='{self._name}', session='{self._session_tag}')"


class SemanticMessageHistory(MessageHistory):
    """Message history with semantic (vector) search over message content.

    Usage:
        hist = SemanticMessageHistory("smart-bot", redis_url="redis://localhost:6379")
        hist.store("Tell me about Redis", "Redis is awesome!")
        results = hist.get_relevant("What is Redis?", vector=[0.1, 0.2, ...], top_k=3)
    """

    _SCHEMA_FIELDS = [
        {"name": "role", "type": "tag"},
        {"name": "content", "type": "text"},
        {"name": "session_tag", "type": "tag"},
        {"name": "timestamp", "type": "numeric"},
        {"name": "vector_field", "type": "vector", "algorithm": "flat",
         "dims": 768, "distance_metric": "cosine"},
    ]

    def __init__(
        self,
        name: str,
        session_tag: str | None = None,
        vector_dims: int = 768,
        distance_threshold: float = 0.3,
        redis_client: Redis | None = None,
        redis_url: str = "redis://localhost:6379",
        overwrite: bool = False,
        **redis_kwargs,
    ):
        self._vector_dims = vector_dims
        self._distance_threshold = distance_threshold

        # Override schema fields with configured dims
        fields = [dict(f) for f in self._SCHEMA_FIELDS]
        for f in fields:
            if f["name"] == "vector_field":
                f["dims"] = vector_dims

        self._schema_fields = fields

        # Call MessageHistory.__init__ but intercept the schema
        self._name = name
        self._session_tag = session_tag or _make_session_tag()
        self._rvl = RedisVL(url=redis_url, **redis_kwargs) if redis_client is None else None

        if redis_client:
            self._rvl = RedisVL.__new__(RedisVL)
            self._rvl._client = redis_client
            self._rvl._url = ""
            self._rvl._redis_kwargs = {}
            self._rvl._schemas = {}

        self._schema = IndexSchema.from_dict({
            "index": {"name": name, "prefix": name, "storage_type": "hash"},
            "fields": self._schema_fields,
        })
        self._rvl.create_index(name, self._schema, overwrite=overwrite)

    def add_message(
        self,
        role: str,
        content: str,
        session_tag: str | None = None,
        vector: list[float] | None = None,
    ) -> list[str]:
        """Store a single message with optional vector."""
        tag = session_tag or self._session_tag
        msg = {
            "role": role,
            "content": content,
            "session_tag": tag,
            "timestamp": time.time(),
        }
        if vector:
            msg["vector_field"] = vector
        return self._rvl.load([msg], self._name, id_field="role")

    def add_messages(
        self,
        messages: list[dict[str, Any]],
        session_tag: str | None = None,
    ) -> list[str]:
        """Batch-add messages. Each dict needs 'role', 'content', optional 'vector'."""
        tag = session_tag or self._session_tag
        ts = time.time()
        docs = []
        for i, m in enumerate(messages):
            doc = {
                "role": m["role"],
                "content": m["content"],
                "session_tag": tag,
                "timestamp": ts + i * 0.001,
            }
            if "vector" in m and m["vector"]:
                doc["vector_field"] = m["vector"]
            docs.append(doc)
        return self._rvl.load(docs, self._name, id_field="role")

    def get_relevant(
        self,
        query_vector: list[float],
        top_k: int = 5,
        session_tag: str | None = None,
        distance_threshold: float | None = None,
        role: str | None = None,
    ) -> list[dict]:
        """Semantic search over message content.

        Args:
            query_vector: The query embedding.
            top_k: Max results.
            session_tag: Filter by session (default: current session).
            distance_threshold: Max cosine distance (0-2).
            role: Optional role filter.
        """
        tag = session_tag or self._session_tag
        expr_parts = []
        if tag:
            expr_parts.append(f"@session_tag:{{{tag}}}")
        if role:
            expr_parts.append(f"@role:{{{role}}}")
        filter_expr = " ".join(expr_parts) if expr_parts else "*"

        threshold = distance_threshold or self._distance_threshold

        from redisvl.query import VectorRangeQuery
        q = VectorRangeQuery(
            vector=query_vector,
            vector_field_name="vector_field",
            distance_threshold=threshold,
            filter_expr=filter_expr,
            return_fields=["role", "content", "session_tag", "timestamp"],
            distance_metric="COSINE",
        )
        return self._rvl.search(q, self._name)

    def set_distance_threshold(self, threshold: float) -> None:
        self._distance_threshold = threshold

    def __repr__(self) -> str:
        return f"SemanticMessageHistory(name='{self._name}', session='{self._session_tag}', threshold={self._distance_threshold})"
