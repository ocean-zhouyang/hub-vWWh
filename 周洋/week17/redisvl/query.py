"""Query builders for Redis vector search.

Each query type produces a Redis Query object (query string + options)
and a params dict that gets passed as query_params to FT.SEARCH.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from redis.commands.search.query import Query as RedisQuery

from redisvl.exceptions import QueryError


class FusionMethod(str, Enum):
    LINEAR = "LINEAR"
    RRF = "RECIPROCAL_RANK"


class TextScorer(str, Enum):
    BM25 = "BM25"
    TFIDF = "TFIDF"
    DISMAX = "DISMAX"


def _normalize_vector(vector: list[float], distance_metric: str) -> list[float]:
    """L2-normalize a vector. Needed for COSINE distance to work correctly."""
    metric = distance_metric.upper()
    if metric not in ("COSINE", "L2"):
        return vector
    arr = np.array(vector, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


def _vector_to_bytes(vector: list[float]) -> bytes:
    """Convert a float list to packed bytes for Redis query params."""
    return np.array(vector, dtype=np.float32).tobytes()


@dataclass
class VectorQuery:
    """K-nearest-neighbor vector similarity search."""

    vector: list[float]
    vector_field_name: str
    top_k: int = 10
    filter_expr: str = "*"
    return_fields: list[str] | None = None
    return_score: bool = True
    distance_metric: str = "COSINE"
    ef_runtime: int | None = None
    dialect: int = 2

    @property
    def params(self) -> dict[str, Any]:
        """Query params for FT.SEARCH."""
        norm_vec = _normalize_vector(self.vector, self.distance_metric)
        p: dict[str, Any] = {"vector": _vector_to_bytes(norm_vec)}
        if self.ef_runtime is not None:
            p["EF_RUNTIME"] = self.ef_runtime
        return p

    def to_redis_query(self) -> RedisQuery:
        """Build the redis-py Query object."""
        if not self.vector:
            raise QueryError("Query vector is empty")

        base = self.filter_expr if self.filter_expr else "*"
        query_str = f"{base}=>[KNN {self.top_k} @{self.vector_field_name} $vector AS vector_distance]"

        q = RedisQuery(query_str).dialect(self.dialect)
        if self.return_fields:
            q.return_fields(*self.return_fields)
        if self.return_score:
            q.with_scores()
        return q

    def set_filter(self, expr: str) -> "VectorQuery":
        self.filter_expr = expr
        return self

    def set_return_fields(self, *fields: str) -> "VectorQuery":
        self.return_fields = list(fields)
        return self


@dataclass
class VectorRangeQuery:
    """Vector search within a distance threshold."""

    vector: list[float]
    vector_field_name: str
    distance_threshold: float
    filter_expr: str = "*"
    return_fields: list[str] | None = None
    return_score: bool = True
    distance_metric: str = "COSINE"
    dialect: int = 2

    @property
    def params(self) -> dict[str, Any]:
        norm_vec = _normalize_vector(self.vector, self.distance_metric)
        return {
            "vector": _vector_to_bytes(norm_vec),
            "distance_threshold": self.distance_threshold,
        }

    def to_redis_query(self) -> RedisQuery:
        query_str = (
            f"@{self.vector_field_name}:[VECTOR_RANGE $distance_threshold $vector]"
            f"=>{{$YIELD_DISTANCE_AS: vector_distance}}"
        )

        q = RedisQuery(query_str).dialect(self.dialect)
        if self.return_fields:
            q.return_fields(*self.return_fields)
        if self.return_score:
            q.with_scores()
        return q


@dataclass
class FilterQuery:
    """Filter-only query (no vector). Full-text or metadata filtering."""

    filter_expr: str = "*"
    return_fields: list[str] | None = None
    top_k: int = 10
    sort_by: str | None = None
    dialect: int = 2

    @property
    def params(self) -> dict[str, Any]:
        return {}

    def to_redis_query(self) -> RedisQuery:
        q = RedisQuery(self.filter_expr).dialect(self.dialect)
        if self.return_fields:
            q.return_fields(*self.return_fields)
        if self.sort_by:
            q.sort_by(self.sort_by)
        q.paging(0, self.top_k)
        return q

    def set_filter(self, expr: str) -> "FilterQuery":
        self.filter_expr = expr
        return self


@dataclass
class TextQuery:
    """Full-text search query."""

    text: str
    text_field: str = "*"
    top_k: int = 10
    return_fields: list[str] | None = None
    dialect: int = 2

    @property
    def params(self) -> dict[str, Any]:
        return {}

    def to_redis_query(self) -> RedisQuery:
        if not self.text.strip():
            raise QueryError("Text query cannot be empty")

        query_str = f"@{self.text_field}:({self.text})"

        q = RedisQuery(query_str).dialect(self.dialect)
        if self.return_fields:
            q.return_fields(*self.return_fields)
        q.paging(0, self.top_k)
        return q


@dataclass
class HybridQuery:
    """Combined text + vector hybrid search. Requires Redis >= 8.4."""

    text: str
    text_field: str
    vector: list[float]
    vector_field: str
    top_k: int = 10
    fusion: FusionMethod = FusionMethod.LINEAR
    alpha: float = 0.5
    rrf_window: int | None = None
    rrf_constant: int | None = None
    return_fields: list[str] | None = None
    dialect: int = 2

    @property
    def params(self) -> dict[str, Any]:
        return {}

    def to_redis_query(self) -> RedisQuery:
        if not self.text.strip():
            raise QueryError("Hybrid query text cannot be empty")
        if not self.vector:
            raise QueryError("Hybrid query vector cannot be empty")

        try:
            from redis.commands.search.query import HybridQuery as RedisHybridQuery
            from redis.commands.search.hybrid_query import (
                CombineResultsMethod,
                HybridSearchQuery,
                HybridVsimQuery,
                ReciprocalRankFusion,
            )
        except ImportError:
            raise QueryError(
                "HybridQuery requires redis-py >= 7.1.0. "
                "Upgrade with: pip install 'redis>=7.1.0'"
            )

        text_query = HybridSearchQuery(f"@{self.text_field}:({self.text})")

        norm_vec = _normalize_vector(self.vector, "COSINE")
        vec_bytes = _vector_to_bytes(norm_vec)
        vector_query = HybridVsimQuery(
            f"@({self.vector_field}:[KNN {self.top_k} $vector])",
            param="vector",
            param_value=vec_bytes,
        )

        if self.fusion == FusionMethod.RRF:
            fusion = ReciprocalRankFusion(
                window=self.rrf_window,
                constant=self.rrf_constant,
            )
        else:
            fusion = CombineResultsMethod.LINEAR

        q = RedisHybridQuery(
            text_query=text_query,
            vector_query=vector_query,
            top_k=self.top_k,
            combination_method=fusion,
            alpha=self.alpha if self.fusion == FusionMethod.LINEAR else None,
        ).dialect(self.dialect)

        if self.return_fields:
            q.return_fields(*self.return_fields)

        return q
