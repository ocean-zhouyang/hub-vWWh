"""Core entry point for Redis Vector Library.

The RedisVL class wraps a Redis connection and provides the full
workflow: create_index -> load -> search, in a clean API surface.
"""

import json
import os
from typing import Any

from redis import Redis
from redis.exceptions import ResponseError

from redisvl.exceptions import IndexError, LoadError, QueryError, SchemaError
from redisvl.loader import DataLoader
from redisvl.query import (
    FilterQuery,
    HybridQuery,
    TextQuery,
    VectorQuery,
    VectorRangeQuery,
)
from redisvl.schema import FieldType, IndexSchema, StorageType


class RedisVL:
    """A crisp, friendly Redis vector search client.

    Usage:
        rvl = RedisVL("redis://localhost:6379")
        rvl.create_index("docs", schema_dict)
        rvl.load(data, "docs")
        results = rvl.vector_search(vec, "docs", top_k=5)

    Args:
        url: Redis connection URL. Falls back to REDIS_URL env var.
        **redis_kwargs: Extra kwargs passed to redis.Redis.from_url().
    """

    def __init__(self, url: str = "", **redis_kwargs: Any):
        self._url = url or os.environ.get("REDIS_URL", "redis://localhost:6379")
        self._redis_kwargs = redis_kwargs
        self._client: Redis | None = None
        self._schemas: dict[str, IndexSchema] = {}

    # ── connection ──────────────────────────────────────────────
    @property
    def client(self) -> Redis:
        """The underlying redis.Redis client (lazy-connects on first use)."""
        if self._client is None:
            self._client = Redis.from_url(self._url, **self._redis_kwargs)
            self._client.ping()
        return self._client

    def disconnect(self) -> None:
        """Close the Redis connection."""
        if self._client:
            self._client.close()
            self._client = None

    # ── index management ────────────────────────────────────────
    def create_index(
        self,
        name: str,
        schema: dict | IndexSchema,
        overwrite: bool = False,
    ) -> None:
        """Create a Redis search index.

        Args:
            name: Unique index name.
            schema: Schema as a dict or IndexSchema instance.
            overwrite: If True, drop and recreate if the index already exists.

        Raises:
            SchemaError: If the schema is invalid.
            IndexError: If index creation fails.
        """
        if isinstance(schema, dict):
            schema = IndexSchema.from_dict(schema)
        schema.index_name = name

        redis_fields = schema.to_redis_fields()
        if not redis_fields:
            raise SchemaError("Schema has no fields defined")

        definition = schema.to_redis_definition()

        try:
            if self._index_exists(name):
                if not overwrite:
                    self._schemas[name] = schema
                    return
                self.drop_index(name)

            self.client.ft(name).create_index(
                fields=redis_fields,
                definition=definition,
            )
            self._schemas[name] = schema
        except ResponseError as e:
            raise IndexError(f"Failed to create index '{name}': {e}") from e

    def drop_index(self, name: str, drop_keys: bool = True) -> None:
        """Delete an index and optionally its associated keys.

        Raises:
            IndexError: If the index does not exist or drop fails.
        """
        if not self._index_exists(name):
            raise IndexError(f"Index '{name}' does not exist")

        try:
            if drop_keys:
                self.client.execute_command("FT.DROPINDEX", name, "DD")
            else:
                self.client.execute_command("FT.DROPINDEX", name)
        except ResponseError as e:
            raise IndexError(f"Failed to drop index '{name}': {e}") from e

        self._schemas.pop(name, None)

    def index_info(self, name: str) -> dict:
        """Get FT.INFO for an index as a human-readable dict.

        Raises:
            IndexError: If the index does not exist.
        """
        if not self._index_exists(name):
            raise IndexError(f"Index '{name}' does not exist")

        try:
            raw = self.client.ft(name).info()
            return _decode_info(raw)
        except ResponseError as e:
            raise IndexError(f"Failed to get info for '{name}': {e}") from e

    def list_indexes(self) -> list[str]:
        """List all search indexes in the connected Redis instance."""
        try:
            result = self.client.execute_command("FT._LIST")
            if result is None:
                return []
            return [r.decode() if isinstance(r, bytes) else r for r in result]
        except ResponseError:
            return []

    # ── data operations ─────────────────────────────────────────
    def load(
        self,
        data: list[dict],
        index_name: str,
        id_field: str | None = None,
        batch_size: int = 200,
    ) -> list[str]:
        """Load documents into an index.

        Args:
            data: List of dict documents.
            index_name: Target index name.
            id_field: Field to use as the document ID.
            batch_size: Documents per pipeline batch.

        Returns:
            List of Redis keys created.

        Raises:
            IndexError: If the schema is not found for the index.
            LoadError: If data validation fails.
        """
        schema = self._get_schema(index_name)
        loader = DataLoader(schema, self.client, batch_size=batch_size)
        return loader.load(data, id_field=id_field)

    def delete(self, ids: list[str], index_name: str) -> int:
        """Delete documents by their IDs from an index.

        Args:
            ids: Document IDs (not full Redis keys).
            index_name: Target index name.

        Returns:
            Number of keys deleted.
        """
        schema = self._get_schema(index_name)
        keys = [schema.key(doc_id) for doc_id in ids]
        return self.client.delete(*keys)

    def clear_index(self, name: str) -> int:
        """Remove all documents from an index but keep the index definition.

        Returns the number of keys removed.
        """
        schema = self._get_schema(name)
        pattern = f"{schema.prefix}{schema.key_separator}*"
        removed = 0
        cursor = 0
        while True:
            cursor, keys = self.client.scan(cursor=cursor, match=pattern, count=500)
            if keys:
                removed += self.client.delete(*keys)
            if cursor == 0:
                break
        return removed

    # ── search ──────────────────────────────────────────────────
    def search(
        self,
        query: VectorQuery | VectorRangeQuery | FilterQuery | TextQuery | HybridQuery,
        index_name: str,
    ) -> list[dict]:
        """Execute a search query against a Redis index.

        Args:
            query: A query object (VectorQuery, TextQuery, etc.).
            index_name: Index to search against.

        Returns:
            List of result dicts with 'id' and any returned fields.
        """
        schema = self._get_schema(index_name)
        redis_q = query.to_redis_query()
        params = getattr(query, "params", {}) or {}

        try:
            raw = self.client.ft(index_name).search(redis_q, query_params=params)
        except ResponseError as e:
            raise QueryError(f"Search failed on '{index_name}': {e}") from e

        return _process_results(raw, schema)

    # ── convenience methods ─────────────────────────────────────
    def vector_search(
        self,
        vector: list[float],
        index_name: str,
        vector_field: str | None = None,
        top_k: int = 10,
        filter_expr: str = "*",
        return_fields: list[str] | None = None,
    ) -> list[dict]:
        """Convenience shortcut for KNN vector search."""
        vf = vector_field or self._detect_vector_field(index_name)
        q = VectorQuery(
            vector=vector,
            vector_field_name=vf,
            top_k=top_k,
            filter_expr=filter_expr,
            return_fields=return_fields,
        )
        return self.search(q, index_name)

    def text_search(
        self,
        text: str,
        index_name: str,
        text_field: str = "*",
        top_k: int = 10,
        return_fields: list[str] | None = None,
    ) -> list[dict]:
        """Convenience shortcut for full-text search."""
        q = TextQuery(
            text=text,
            text_field=text_field,
            top_k=top_k,
            return_fields=return_fields,
        )
        return self.search(q, index_name)

    def filter_search(
        self,
        filter_expr: str,
        index_name: str,
        top_k: int = 10,
        return_fields: list[str] | None = None,
        sort_by: str | None = None,
    ) -> list[dict]:
        """Convenience shortcut for filter queries."""
        q = FilterQuery(
            filter_expr=filter_expr,
            top_k=top_k,
            return_fields=return_fields,
            sort_by=sort_by,
        )
        return self.search(q, index_name)

    def hybrid_search(
        self,
        text: str,
        text_field: str,
        vector: list[float],
        vector_field: str,
        index_name: str,
        top_k: int = 10,
        fusion: str = "LINEAR",
        alpha: float = 0.5,
        return_fields: list[str] | None = None,
    ) -> list[dict]:
        """Convenience shortcut for hybrid text+vector search."""
        from redisvl.query import FusionMethod

        q = HybridQuery(
            text=text,
            text_field=text_field,
            vector=vector,
            vector_field=vector_field,
            top_k=top_k,
            fusion=FusionMethod(fusion),
            alpha=alpha,
            return_fields=return_fields,
        )
        return self.search(q, index_name)

    # ── internal helpers ────────────────────────────────────────
    def _index_exists(self, name: str) -> bool:
        try:
            indexes = self.list_indexes()
            return name in indexes
        except Exception:
            return False

    def _get_schema(self, index_name: str) -> IndexSchema:
        """Get the registered schema for an index, or try to infer it."""
        if index_name in self._schemas:
            return self._schemas[index_name]

        # Try to infer from FT.INFO
        try:
            info = self.index_info(index_name)
            schema = IndexSchema.from_dict(info)
            self._schemas[index_name] = schema
            return schema
        except Exception:
            raise IndexError(
                f"No schema registered for index '{index_name}'. "
                "Create it first with create_index() or use an existing Redis index."
            )

    def _detect_vector_field(self, index_name: str) -> str:
        """Find the name of a vector field in the schema."""
        schema = self._get_schema(index_name)
        vf = schema.get_vector_field()
        if vf is None:
            raise QueryError(f"No vector field found in index '{index_name}'")
        return vf.name

    def __repr__(self) -> str:
        connected = "connected" if self._client else "disconnected"
        return f"RedisVL({self._url}, {connected})"

    def __enter__(self) -> "RedisVL":
        return self

    def __exit__(self, *args: Any) -> None:
        self.disconnect()


# ── result processing ────────────────────────────────────────────
def _process_results(raw, schema: IndexSchema) -> list[dict]:
    """Process raw FT.SEARCH results into clean dicts.

    - Decodes byte strings
    - Unpacks JSON-encoded values for JSON storage
    - Strips the 'payload' field
    - Preserves 'id' and 'vector_distance'
    """
    results = []
    for doc in raw.docs:
        d = {}
        for key, value in doc.__dict__.items():
            if key == "payload":
                continue
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            d[key] = value

        # For JSON storage, the full doc might be under 'json'
        if schema.storage_type == StorageType.JSON and "json" in d:
            json_data = d.pop("json")
            if isinstance(json_data, str):
                try:
                    json_data = json.loads(json_data)
                except json.JSONDecodeError:
                    pass
            if isinstance(json_data, dict):
                # Merge: 'id' from doc takes priority
                merged = {**json_data, "id": d.get("id")}
                results.append(merged)
                continue

        results.append(d)

    return results


def _decode_info(raw: dict) -> dict:
    """Decode byte keys/values in FT.INFO output."""
    result = {}
    for key, value in raw.items():
        k = key.decode() if isinstance(key, bytes) else key
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        elif isinstance(value, list):
            value = [
                v.decode() if isinstance(v, bytes) else v
                for v in value
            ]
        elif isinstance(value, dict):
            value = _decode_info(value)
        result[k] = value
    return result
