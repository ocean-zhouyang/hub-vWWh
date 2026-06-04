"""Semantic Router — classify text by semantic similarity to reference phrases.

Define routes with example references, then classify new input by finding
which route's references it is most similar to.
"""

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from redis import Redis

from redisvl.client import RedisVL
from redisvl.query import VectorRangeQuery
from redisvl.schema import IndexSchema


class AggregationMethod(str, Enum):
    MIN = "min"
    AVG = "avg"


@dataclass
class Route:
    """A named route with reference phrases for matching.

    Args:
        name: Unique route name.
        references: List of example phrases that should match this route.
        distance_threshold: Max cosine distance for this route (0-2).
        metadata: Optional extra data attached to the route.
    """
    name: str
    references: list[str]
    distance_threshold: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteMatch:
    """Result of routing a statement."""
    name: str | None = None
    distance: float | None = None
    metadata: dict[str, Any] | None = None

    @property
    def matched(self) -> bool:
        return self.name is not None


def _ref_hash(reference: str) -> str:
    return hashlib.sha256(reference.encode()).hexdigest()[:12]


class SemanticRouter:
    """Route inputs to predefined categories by semantic similarity.

    Usage:
        routes = [
            Route("greeting", ["hello", "hi there", "good morning"], 0.3),
            Route("billing",  ["invoice", "payment", "charge me"], 0.4),
            Route("support",  ["it's broken", "help me", "not working"], 0.35),
        ]
        router = SemanticRouter("my-router", routes)
        result = router("hey how are you", vector=[0.1, 0.2, ...])
        print(result.name)  # -> "greeting"
    """

    _SCHEMA_FIELDS = [
        {"name": "route_name", "type": "tag"},
        {"name": "reference", "type": "text"},
        {"name": "vector", "type": "vector", "algorithm": "flat",
         "dims": 768, "distance_metric": "cosine"},
    ]

    def __init__(
        self,
        name: str,
        routes: list[Route] | None = None,
        vector_dims: int = 768,
        default_threshold: float = 0.5,
        aggregation: AggregationMethod = AggregationMethod.MIN,
        redis_client: Redis | None = None,
        redis_url: str = "redis://localhost:6379",
        overwrite: bool = False,
        **redis_kwargs,
    ):
        self._name = name
        self._default_threshold = default_threshold
        self._aggregation = aggregation
        self._routes: dict[str, Route] = {}

        fields = [dict(f) for f in self._SCHEMA_FIELDS]
        for f in fields:
            if f["name"] == "vector":
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

        if routes:
            for route in routes:
                self.add_route(route)

    # ── route management ───────────────────────────────────────
    def add_route(self, route: Route, vectors: list[list[float]] | None = None) -> None:
        """Add a route and index its references.

        Args:
            route: The Route to add.
            vectors: Pre-computed embeddings for each reference.
                     If not provided, references are stored without vectors
                     (text-only matching via full-text search).
        """
        self._routes[route.name] = route
        docs = []
        for i, ref in enumerate(route.references):
            doc = {
                "entry_id": _ref_hash(f"{route.name}:{ref}"),
                "route_name": route.name,
                "reference": ref,
            }
            if vectors and i < len(vectors):
                doc["vector"] = vectors[i]
            docs.append(doc)
        self._rvl.load(docs, self._name, id_field="entry_id")

    def remove_route(self, name: str) -> bool:
        """Remove a route and its references. Returns True if found."""
        if name not in self._routes:
            return False
        del self._routes[name]
        # Remove all docs for this route
        pattern = f"{self._name}:*"
        cursor = 0
        while True:
            cursor, keys = self._rvl.client.scan(cursor=cursor, match=pattern, count=100)
            for key in keys:
                raw = self._rvl.client.hget(key, "route_name")
                if raw:
                    rn = raw.decode() if isinstance(raw, bytes) else raw
                    if rn == name:
                        self._rvl.client.delete(key)
            if cursor == 0:
                break
        return True

    def get_route(self, name: str) -> Route | None:
        """Get a route by name."""
        return self._routes.get(name)

    def list_routes(self) -> list[str]:
        """List all route names."""
        return list(self._routes.keys())

    # ── routing ────────────────────────────────────────────────
    def __call__(
        self,
        statement: str,
        vector: list[float],
        aggregation: AggregationMethod | None = None,
    ) -> RouteMatch:
        """Classify a statement to a single best route.

        Args:
            statement: The input text (used for fallback text search).
            vector: The statement's embedding.
            aggregation: Override default aggregation method.

        Returns:
            RouteMatch with the best-matching route. name is None if no match.
        """
        matches = self.route_many(statement, vector, max_k=1, aggregation=aggregation)
        if matches:
            return matches[0]
        return RouteMatch()

    def route_many(
        self,
        statement: str,
        vector: list[float],
        max_k: int = 3,
        aggregation: AggregationMethod | None = None,
        fallback_threshold: float | None = None,
    ) -> list[RouteMatch]:
        """Classify a statement to multiple possible routes.

        Args:
            statement: The input text.
            vector: The statement's embedding.
            max_k: Max routes to return.
            aggregation: Override aggregation method.
            fallback_threshold: Threshold for routes without explicit threshold.

        Returns:
            List of RouteMatch sorted by distance (closest first).
        """
        agg = aggregation or self._aggregation
        fallback = fallback_threshold or self._default_threshold

        # Collect per-route distances from matching references
        route_distances: dict[str, list[float]] = {}

        # For each route, search its references within the route's threshold
        for route_name, route in self._routes.items():
            threshold = route.distance_threshold if route.distance_threshold else fallback
            try:
                q = VectorRangeQuery(
                    vector=vector,
                    vector_field_name="vector",
                    distance_threshold=threshold,
                    filter_expr=f"@route_name:{{{route_name}}}",
                    return_fields=["route_name"],
                    distance_metric="COSINE",
                )
                results = self._rvl.search(q, self._name)
                if results:
                    distances = [float(r.get("vector_distance", 1.0)) for r in results]
                    route_distances[route_name] = distances
            except Exception:
                continue

        if not route_distances:
            return []

        # Aggregate per route
        scored: list[tuple[str, float, dict[str, Any]]] = []
        for route_name, distances in route_distances.items():
            if agg == AggregationMethod.MIN:
                score = min(distances)
            else:
                score = sum(distances) / len(distances)
            route = self._routes[route_name]
            threshold = route.distance_threshold or fallback
            if score <= threshold:
                scored.append((route_name, score, route.metadata))

        # Sort by distance ascending
        scored.sort(key=lambda x: x[1])

        return [
            RouteMatch(name=name, distance=dist, metadata=meta)
            for name, dist, meta in scored[:max_k]
        ]

    # ── maintenance ────────────────────────────────────────────
    def delete(self) -> None:
        """Delete the index and all route references."""
        self._rvl.drop_index(self._name)
        self._rvl.disconnect()

    def clear(self) -> int:
        """Remove all route references but keep the index."""
        return self._rvl.clear_index(self._name)

    def disconnect(self) -> None:
        self._rvl.disconnect()

    def __repr__(self) -> str:
        routes = ", ".join(self._routes.keys())
        return f"SemanticRouter(name='{self._name}', routes=[{routes}])"
