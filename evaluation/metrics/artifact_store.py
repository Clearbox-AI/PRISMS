from typing import Any, Dict

class ArtifactStore:
    """
    A simple store for caching data/models across multiple metrics.

    # TODO:
    - Check if a file already exists on disk, load it, store it, etc.
    - Implement a TTL (time-to-live) or versioning to force regeneration.
    """

    def __init__(self):
        self._in_memory_cache: Dict[str, Any] = {}

    def get(self, key: str) -> Any:
        """
        Retrieve an artifact by key, or None if missing.
        """
        return self._in_memory_cache.get(key, None)

    def put(self, key: str, value: Any) -> None:
        """
        Store an artifact under a given key.
        """
        self._in_memory_cache[key] = value

    def has(self, key: str) -> bool:
        """
        Check if the key is already in the store.
        """
        return key in self._in_memory_cache

    def clear(self) -> None:
        """
        Clear all stored artifacts.
        """
        self._in_memory_cache.clear()