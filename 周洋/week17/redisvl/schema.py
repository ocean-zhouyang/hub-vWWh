"""Schema definitions for Redis Vector Library."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from redis.commands.search.field import (
    Field as RedisField,
    NumericField as RedisNumericField,
    TagField as RedisTagField,
    TextField as RedisTextField,
    VectorField as RedisVectorField,
)
from redis.commands.search.index_definition import IndexDefinition, IndexType

from redisvl.exceptions import SchemaError


class StorageType(str, Enum):
    HASH = "hash"
    JSON = "json"


class FieldType(str, Enum):
    TEXT = "text"
    TAG = "tag"
    NUMERIC = "numeric"
    VECTOR = "vector"


class VectorAlgorithm(str, Enum):
    FLAT = "flat"
    HNSW = "hnsw"


class VectorDistance(str, Enum):
    COSINE = "cosine"
    L2 = "l2"
    IP = "ip"


class VectorDataType(str, Enum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"


@dataclass
class FieldDef:
    """Definition for a single field in the index schema."""

    name: str
    type: FieldType
    attrs: dict[str, Any] = field(default_factory=dict)
    path: str | None = None

    def to_redis_field(self) -> RedisField:
        """Convert to a redis-py Field object ready for FT.CREATE."""
        match self.type:
            case FieldType.TEXT:
                kwargs: dict[str, Any] = {"name": self.name}
                if self.attrs.get("weight"):
                    kwargs["weight"] = self.attrs["weight"]
                if self.attrs.get("no_stem"):
                    kwargs["no_stem"] = self.attrs["no_stem"]
                if self.attrs.get("sortable"):
                    kwargs["sortable"] = True
                if self.path:
                    kwargs["path"] = self.path
                return RedisTextField(**kwargs)

            case FieldType.TAG:
                kwargs = {"name": self.name}
                if self.attrs.get("separator"):
                    kwargs["separator"] = self.attrs["separator"]
                if self.attrs.get("case_sensitive"):
                    kwargs["case_sensitive"] = self.attrs["case_sensitive"]
                if self.attrs.get("sortable"):
                    kwargs["sortable"] = True
                if self.path:
                    kwargs["path"] = self.path
                return RedisTagField(**kwargs)

            case FieldType.NUMERIC:
                kwargs = {"name": self.name}
                if self.attrs.get("sortable"):
                    kwargs["sortable"] = True
                if self.path:
                    kwargs["path"] = self.path
                return RedisNumericField(**kwargs)

            case FieldType.VECTOR:
                algorithm = self.attrs.get("algorithm", "flat").lower()
                dims = self.attrs.get("dims")
                if dims is None:
                    raise SchemaError(f"Vector field '{self.name}' requires 'dims'")
                distance_metric = self.attrs.get("distance_metric", "COSINE").upper()
                datatype = self.attrs.get("datatype", "FLOAT32").upper()

                attributes: dict[str, Any] = {
                    "TYPE": datatype,
                    "DIM": dims,
                    "DISTANCE_METRIC": distance_metric,
                }

                if algorithm == "hnsw":
                    attributes["M"] = self.attrs.get("m", 16)
                    attributes["EF_CONSTRUCTION"] = self.attrs.get("ef_construction", 200)
                    attributes["EF_RUNTIME"] = self.attrs.get("ef_runtime", 10)
                    if "epsilon" in self.attrs:
                        attributes["EPSILON"] = self.attrs["epsilon"]

                if self.attrs.get("initial_cap"):
                    attributes["INITIAL_CAP"] = self.attrs["initial_cap"]

                return RedisVectorField(
                    name=self.name,
                    algorithm=algorithm.upper(),
                    attributes=attributes,
                )

            case _:
                raise SchemaError(f"Unknown field type: {self.type}")


@dataclass
class IndexSchema:
    """Schema for a Redis search index.

    Can be created from a YAML file or Python dictionary.

    Example YAML:
        index:
          name: products
          prefix: prod
          storage_type: json
        fields:
          - name: title
            type: text
          - name: embedding
            type: vector
            algorithm: hnsw
            dims: 768
            distance_metric: cosine
    """

    index_name: str
    prefix: str = "rvl"
    key_separator: str = ":"
    storage_type: StorageType = StorageType.HASH
    fields: list[FieldDef] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "IndexSchema":
        """Create an IndexSchema from a dictionary."""
        index_data = data.get("index", data)
        if isinstance(index_data, dict):
            name = index_data.get("name", "")
            if not name:
                raise SchemaError("Index name is required")
            prefix = index_data.get("prefix", "rvl")
            key_sep = index_data.get("key_separator", ":")
            storage = StorageType(index_data.get("storage_type", "hash"))
        else:
            raise SchemaError("Schema dict must have an 'index' section")

        field_defs = []
        for f in data.get("fields", []):
            ftype = FieldType(f["type"])
            attrs = {k: v for k, v in f.items() if k not in ("name", "type", "path")}
            field_defs.append(FieldDef(name=f["name"], type=ftype, attrs=attrs, path=f.get("path")))

        return cls(
            index_name=name,
            prefix=prefix,
            key_separator=key_sep,
            storage_type=storage,
            fields=field_defs,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "IndexSchema":
        """Create an IndexSchema from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise SchemaError(f"Schema file not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Serialize back to a dict."""
        fields_data = []
        for f in self.fields:
            fd = {"name": f.name, "type": f.type.value, **f.attrs}
            if f.path:
                fd["path"] = f.path
            fields_data.append(fd)

        return {
            "index": {
                "name": self.index_name,
                "prefix": self.prefix,
                "key_separator": self.key_separator,
                "storage_type": self.storage_type.value,
            },
            "fields": fields_data,
        }

    def to_redis_fields(self) -> list[RedisField]:
        """Generate the redis-py Field list for FT.CREATE."""
        return [f.to_redis_field() for f in self.fields]

    def to_redis_definition(self) -> IndexDefinition:
        """Generate the IndexDefinition for FT.CREATE."""
        idx_type = IndexType.JSON if self.storage_type == StorageType.JSON else IndexType.HASH
        return IndexDefinition(prefix=[self.prefix], index_type=idx_type)

    def key(self, doc_id: str) -> str:
        """Construct a Redis key: prefix + separator + id."""
        return f"{self.prefix}{self.key_separator}{doc_id}"

    def add_field(self, field_def: FieldDef) -> None:
        """Add a field to the schema."""
        self.fields.append(field_def)

    def get_vector_field(self) -> FieldDef | None:
        """Return the first vector field found in the schema."""
        for f in self.fields:
            if f.type == FieldType.VECTOR:
                return f
        return None
