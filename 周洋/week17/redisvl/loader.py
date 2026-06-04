"""Data loader for Redis Vector Library.

Handles batch loading of documents into Redis, supporting both
HASH and JSON storage types.
"""

import json
import uuid
from typing import Any

from redis import Redis

from redisvl.exceptions import LoadError
from redisvl.schema import FieldType, IndexSchema, StorageType


def _generate_id() -> str:
    """Generate a unique ID using timestamp + uuid4 hex.

    Fast, unique enough for most workloads, no extra dependencies.
    """
    return uuid.uuid4().hex[:16]


class DataLoader:
    """Load data into a Redis search index.

    Args:
        schema: The index schema defining the fields and storage type.
        client: A redis.Redis client instance.
        batch_size: Number of documents per pipeline batch.
    """

    def __init__(self, schema: IndexSchema, client: Redis, batch_size: int = 200):
        self.schema = schema
        self.client = client
        self.batch_size = batch_size

    def load(self, data: list[dict], id_field: str | None = None) -> list[str]:
        """Load a list of documents into Redis.

        Args:
            data: List of dicts representing documents.
            id_field: Field to use as the document ID. If None,
                      looks for "id" field, otherwise auto-generates.

        Returns:
            List of Redis keys that were created.
        """
        if not data:
            return []

        keys = []
        pipe = self.client.pipeline(transaction=False)
        count = 0

        for i, doc in enumerate(data):
            doc_id = self._resolve_id(doc, id_field)
            key = self.schema.key(doc_id)

            if self.schema.storage_type == StorageType.JSON:
                pipe.json().set(key, "$", doc)
            else:
                # HASH: flatten nested values as JSON strings
                flat = self._flatten_for_hash(doc)
                pipe.hset(key, mapping=flat)

            keys.append(key)
            count += 1

            if count % self.batch_size == 0:
                pipe.execute()
                pipe = self.client.pipeline(transaction=False)

        # Execute remaining
        if count % self.batch_size != 0:
            pipe.execute()

        return keys

    def load_dataframe(self, df, id_field: str | None = None) -> list[str]:
        """Load a pandas DataFrame into Redis.

        Pandas is imported lazily -- only needed when using this method.
        """
        try:
            data = df.to_dict(orient="records")
        except ImportError:
            raise LoadError("pandas is required for load_dataframe(). Install with: pip install pandas")
        return self.load(data, id_field=id_field)

    def delete(self, keys: list[str]) -> int:
        """Delete documents by their Redis keys. Returns count deleted."""
        if not keys:
            return 0
        return self.client.delete(*keys)

    def _resolve_id(self, doc: dict, id_field: str | None) -> str:
        """Resolve the document ID from the document itself or generate one."""
        if id_field and id_field in doc:
            return str(doc[id_field])
        if "id" in doc:
            return str(doc["id"])
        return _generate_id()

    @staticmethod
    def _flatten_for_hash(doc: dict) -> dict[str, str]:
        """Flatten a nested dict for Redis HASH storage.

        Nested dicts/lists get serialized to JSON strings.
        Flat values are stringified as-is.
        """
        flat = {}
        for key, value in doc.items():
            if value is None:
                flat[key] = ""
            elif isinstance(value, (dict, list)):
                flat[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, (int, float)):
                flat[key] = str(value)
            elif isinstance(value, str):
                flat[key] = value
            else:
                flat[key] = json.dumps(value, ensure_ascii=False, default=str)
        return flat

    def _validate_doc(self, doc: dict, index: int) -> None:
        """Validate a document against the schema.

        Checks that vector fields have the correct dimension.
        """
        for field_def in self.schema.fields:
            if field_def.type == FieldType.VECTOR:
                dims = field_def.attrs.get("dims")
                if dims is None:
                    continue
                if field_def.name in doc:
                    value = doc[field_def.name]
                    if isinstance(value, list) and len(value) != dims:
                        raise LoadError(
                            f"Document {index}: vector field '{field_def.name}' "
                            f"expects {dims} dimensions, got {len(value)}"
                        )
