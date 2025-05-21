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





import os
import shutil
import tempfile
import pickle

def test_in_memory_store():
    """
    Test that storing/loading artifacts in memory works without a disk path.
    """
    print("[test_in_memory_store] Starting...")

    store = ArtifactStore(artifact_root=None)
    test_key = "test_key_in_memory"
    test_value = {"example": 123}

    # The store should not have the artifact initially
    assert not store.has_artifact(test_key), "Store shouldn't have the artifact yet."
    assert store.get_artifact(test_key) is None, "Should get None for missing artifact."

    # Put artifact into the store (in-memory only)
    store.put_artifact(test_key, test_value, save_to_disk=False)

    # Now it should be present
    assert store.has_artifact(test_key), "Artifact should now exist in store."
    retrieved = store.get_artifact(test_key)
    assert retrieved == test_value, "Retrieved value does not match stored value."

    print("[test_in_memory_store] Passed.")

def test_disk_store():
    """
    Test that storing/loading artifacts on disk works when artifact_root is provided.
    """
    print("[test_disk_store] Starting...")
    tmp_dir = tempfile.mkdtemp(prefix="artifact_store_test_")

    try:
        store = ArtifactStore(artifact_root=tmp_dir)
        test_key = "test_key_on_disk"
        test_value = [1, 2, 3]

        # Put the artifact and save to disk
        store.put_artifact(test_key, test_value, save_to_disk=True)
        assert store.has_artifact(test_key), "Store should have the artifact in memory."

        # Clear the in-memory cache to simulate a fresh environment
        store._in_memory_cache = {}
        assert not store.has_artifact(test_key), "Cache was cleared; artifact should be absent in memory."

        # Now load from disk
        loaded_value = store.load_artifact(test_key)
        assert loaded_value == test_value, "Loaded artifact from disk doesn't match original."
        # Once loaded, it should also be in memory
        assert store.has_artifact(test_key), "Artifact should be back in memory after loading."

        print("[test_disk_store] Passed.")
    finally:
        # Cleanup the temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)

def test_load_artifact_missing_key():
    """
    Check behavior if we try to load an artifact that doesn't exist.
    """
    print("[test_load_artifact_missing_key] Starting...")
    tmp_dir = tempfile.mkdtemp(prefix="artifact_store_test_")
    try:
        store = ArtifactStore(artifact_root=tmp_dir)
        missing_key = "missing_key"
        # No artifact with 'missing_key' put yet, so should return None
        result = store.load_artifact(missing_key)
        assert result is None, "Loading a non-existent artifact should return None."
        print("[test_load_artifact_missing_key] Passed.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_disk_corruption():
    """
    Check that we handle disk-load exceptions gracefully (e.g. file corruption).
    """
    print("[test_disk_corruption] Starting...")
    tmp_dir = tempfile.mkdtemp(prefix="artifact_store_test_")
    try:
        store = ArtifactStore(artifact_root=tmp_dir)
        test_key = "test_corrupt"
        test_value = {"hello": "world"}
        store.put_artifact(test_key, test_value, save_to_disk=True)

        # Corrupt the file after saving
        corrupt_path = os.path.join(tmp_dir, f"{test_key}.pkl")
        with open(corrupt_path, "wb") as f:
            f.write(b"not a valid pickle file")

        # Clear in-memory cache
        store._in_memory_cache = {}

        # Attempt to load the corrupted artifact
        loaded_value = store.load_artifact(test_key)
        # Because it's corrupted, load_artifact should return None
        assert loaded_value is None, "Loading a corrupted artifact should return None."
        assert not store.has_artifact(test_key), "After failed load, artifact shouldn't exist in memory."

        print("[test_disk_corruption] Passed.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def main():
    """
    Entry point to run all artifact store tests from a single script.
    """
    print("===== Running ArtifactStore tests from main() =====")
    test_in_memory_store()
    test_disk_store()
    test_load_artifact_missing_key()
    test_disk_corruption()
    print("===== All tests passed! =====")


if __name__ == "__main__":
    main()