"""RedisVL -- A crisp, friendly Redis vector search library."""

from redisvl.client import RedisVL
from redisvl.extensions import (
    EmbeddingsCache,
    MessageHistory,
    Route,
    RouteMatch,
    SemanticCache,
    SemanticMessageHistory,
    SemanticRouter,
)
from redisvl.query import FilterQuery, HybridQuery, TextQuery, VectorQuery
from redisvl.schema import IndexSchema

__version__ = "0.1.0"

__all__ = [
    "RedisVL",
    "IndexSchema",
    "VectorQuery",
    "TextQuery",
    "FilterQuery",
    "HybridQuery",
    # Extensions
    "SemanticCache",
    "EmbeddingsCache",
    "MessageHistory",
    "SemanticMessageHistory",
    "SemanticRouter",
    "Route",
    "RouteMatch",
]
