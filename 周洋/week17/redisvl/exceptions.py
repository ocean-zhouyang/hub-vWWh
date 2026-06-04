class RedisVLError(Exception):
    """All redisvl errors inherit from here."""
    pass


class SchemaError(RedisVLError):
    """Schema validation or parsing problems."""
    pass


class IndexError(RedisVLError):
    """Index creation, deletion, or info failures."""
    pass


class QueryError(RedisVLError):
    """Query construction or execution failures."""
    pass


class LoadError(RedisVLError):
    """Data loading or validation failures."""
    pass
