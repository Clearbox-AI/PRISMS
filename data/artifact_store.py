import os
import pickle
from typing import Any, Dict, Optional, Callable


# class ArtifactStore:
#     """
#     A store for caching or retrieving various artifacts, with in-memory caching
#     and optional disk-based pickling. We also demonstrate cross-session usage
#     by attempting to load from disk if not found in memory.
#     """
#
#     def __init__(self, artifact_root: Optional[str] = None):
#         """
#         :param artifact_root: Directory in which artifact .pkl files will be stored.
#                               If None, on-disk saving/loading is disabled.
#         """
#         self.artifact_root = artifact_root
#         self._in_memory_cache: Dict[str, Any] = {}
#
#         if self.artifact_root:
#             os.makedirs(self.artifact_root, exist_ok=True)
#
#     def has_artifact(self, key: str) -> bool:
#         return key in self._in_memory_cache
#
#     def get_artifact(self, key: str) -> Any:
#         return self._in_memory_cache.get(key, None)
#
#     def put_artifact(self, key: str, value: Any, save_to_disk: bool = False) -> None:
#         self._in_memory_cache[key] = value
#         if save_to_disk and self.artifact_root is not None:
#             self._save_to_disk(key, value)
#
#     def get_or_create_artifact(
#         self,
#         key: str,
#         creator_fn: Callable[[], Any],
#         force: bool = False,
#         save_to_disk: bool = False
#     ) -> Any:
#         """
#         If artifact not in memory or 'force=True', calls `creator_fn()` to create it,
#         then stores it (and optionally pickles it).
#         If not forcing, also attempts to load from disk if found.
#         """
#         # 1) If not forcing, check memory
#         if not force and self.has_artifact(key):
#             return self.get_artifact(key)
#
#         # 2) If not forcing, try loading from disk
#         if not force and self.artifact_root:
#             loaded_val = self._load_from_disk(key)
#             if loaded_val is not None:
#                 self._in_memory_cache[key] = loaded_val
#                 return loaded_val
#
#         # 3) Otherwise, create
#         value = creator_fn()
#         self.put_artifact(key, value, save_to_disk=save_to_disk)
#         return value
#
#     def clear(self) -> None:
#         """Clear only the in-memory cache."""
#         self._in_memory_cache.clear()
#
#     def _save_to_disk(self, key: str, value: Any):
#         filename = self._artifact_filename(key)
#         try:
#             with open(filename, "wb") as f:
#                 pickle.dump(value, f)
#         except Exception as e:
#             print(f"[ArtifactStore Warning] Could not save artifact '{key}': {e}")
#
#     def _load_from_disk(self, key: str) -> Any:
#         filename = self._artifact_filename(key)
#         if os.path.exists(filename):
#             try:
#                 with open(filename, "rb") as f:
#                     return pickle.load(f)
#             except Exception as e:
#                 print(f"[ArtifactStore Warning] Could not load artifact '{key}': {e}")
#         return None
#
#     def _artifact_filename(self, key: str) -> str:
#         if not self.artifact_root:
#             return key  # no disk usage if root not set
#         return os.path.join(self.artifact_root, f"{key}.pkl")


class ArtifactStore:
    """
    A store for caching artifacts (like data or models) in memory, optionally
    pickling them to disk for cross-session usage.
    """

    def __init__(self, artifact_root: Optional[str] = None):
        self.artifact_root = artifact_root
        self._in_memory_cache: Dict[str, Any] = {}
        if artifact_root:
            os.makedirs(artifact_root, exist_ok=True)

    def has_artifact(self, key: str) -> bool:
        return key in self._in_memory_cache

    def get_artifact(self, key: str) -> Any:
        return self._in_memory_cache.get(key, None)

    def put_artifact(self, key: str, value: Any, save_to_disk: bool = False) -> None:
        """
        Save in memory and optionally pickle to <artifact_root>/<key>.pkl.
        """
        self._in_memory_cache[key] = value
        if save_to_disk and self.artifact_root:
            path = os.path.join(self.artifact_root, f"{key}.pkl")
            try:
                with open(path, "wb") as f:
                    pickle.dump(value, f)
            except Exception as e:
                print(f"[ArtifactStore] Could not save {key}: {e}")

    def load_artifact(self, key: str) -> Any:
        """
        If not in memory, attempt to load from <artifact_root>/<key>.pkl
        """
        if key in self._in_memory_cache:
            return self._in_memory_cache[key]

        if not self.artifact_root:
            return None

        path = os.path.join(self.artifact_root, f"{key}.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    val = pickle.load(f)
                self._in_memory_cache[key] = val
                return val
            except Exception as e:
                print(f"[ArtifactStore] Could not load {key} from disk: {e}")
        return None