import os
import pickle
from typing import Any, Dict, Callable

class ArtifactStore:
    """
    A store for caching or retrieving various artifacts used by metrics:
      - In-memory caching
      - (Optional) on-disk persistence
      - Utility methods to "get or create" items to avoid re-doing heavy work

    Typical usage:
      - store = ArtifactStore(config)
      - real_images = store.get_or_create_artifact("real_images", creator_fn=load_real_images)
      - synthetic_images = store.get_or_create_artifact("synth_images", creator_fn=generate_synth_images)
      - encoder = store.get_or_create_pretrained_encoder(...)
      - model = store.get_or_create_trained_model(...)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        :param config: A config dict that can store paths, etc.
                       e.g. config["artifact_root"] to specify where to store files
        """
        self.config = config
        self._in_memory_cache: Dict[str, Any] = {}

        # possible on-disk caching:
        self.artifact_root = config.get("artifact_root", None)
        if self.artifact_root and not os.path.exists(self.artifact_root):
            os.makedirs(self.artifact_root, exist_ok=True)

    def has_artifact(self, key: str) -> bool:
        """Check if the artifact is in memory cache."""
        return key in self._in_memory_cache

    def get_artifact(self, key: str) -> Any:
        """Retrieve an artifact from in-memory cache. Returns None if missing."""
        return self._in_memory_cache.get(key, None)

    def put_artifact(self, key: str, value: Any, save_to_disk: bool = False) -> None:
        """
        Store an artifact in memory (and optionally on disk).
        """
        self._in_memory_cache[key] = value
        if save_to_disk and self.artifact_root is not None:
            self._save_to_disk(key, value)

    def get_or_create_artifact(
        self,
        key: str,
        creator_fn: Callable[[], Any],
        force: bool = False,
        save_to_disk: bool = False
    ) -> Any:
        """
        Checks whether artifact is already in store. If not (or `force=True`),
        calls `creator_fn()` to create it, then stores it.
        Returns the artifact.
        """
        # If not forcing and we have it in memory, return
        if not force and self.has_artifact(key):
            return self.get_artifact(key)

        # Otherwise, create and store
        value = creator_fn()
        self.put_artifact(key, value, save_to_disk=save_to_disk)
        return value

    def clear(self) -> None:
        """Clear all in-memory artifacts."""
        # TODO: ok clearing the dict, but also deleting from disk correspondent files
        self._in_memory_cache.clear()

    # --------------------------------------------------------------------------
    # Below are optional "helper" methods you might define for common tasks
    # so that the code is less repetitive in the metrics themselves.
    # --------------------------------------------------------------------------

    def get_or_create_pretrained_encoder(
        self,
        key: str,
        encoder_factory: Callable[[], Any],
        force: bool = False,
        save_to_disk: bool = False
    ) -> Any:
        """
        Retrieve or create a pretrained encoder (like Inception).
        E.g. usage:
          encoder = store.get_or_create_pretrained_encoder(
               key="inception_v3",
               encoder_factory=load_inception_v3,
               force=False
          )
        """
        return self.get_or_create_artifact(
            key=key,
            creator_fn=encoder_factory,
            force=force,
            save_to_disk=save_to_disk
        )

    def get_or_create_trained_model(
        self,
        key: str,
        training_fn: Callable[[], Any],
        force: bool = False,
        save_to_disk: bool = False
    ) -> Any:
        """
        Retrieve or train a model, storing it under `key`.
        E.g. usage:
          model = store.get_or_create_trained_model(
               key="attribute_model",
               training_fn=train_attribute_model,
               force=False
          )
        """
        return self.get_or_create_artifact(
            key=key,
            creator_fn=training_fn,
            force=force,
            save_to_disk=save_to_disk
        )

    def get_or_create_merged_dataloader(
        self,
        key: str,
        merge_fn: Callable[[], Any],
        force: bool = False,
        save_to_disk: bool = False
    ) -> Any:
        """
        Retrieve or create a merged DataLoader for real+synthetic data.
        E.g. usage:
          merged_dl = store.get_or_create_merged_dataloader(
              key="merged_real_synth",
              merge_fn=create_merged_dl,
              force=False
          )
        """
        return self.get_or_create_artifact(
            key=key,
            creator_fn=merge_fn,
            force=force,
            save_to_disk=save_to_disk
        )

    # --------------------------------------------------------------------------
    # Optional: On-disk save/load to persist artifacts between runs
    # --------------------------------------------------------------------------
    def _save_to_disk(self, key: str, value: Any):
        filename = self._artifact_filename(key)
        try:
            with open(filename, "wb") as f:
                pickle.dump(value, f)
        except Exception as e:
            print(f"Warning: could not save artifact '{key}' to disk: {e}")

    def _load_from_disk(self, key: str):
        filename = self._artifact_filename(key)
        if os.path.exists(filename):
            try:
                with open(filename, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: could not load artifact '{key}' from disk: {e}")
        return None

    def _artifact_filename(self, key: str) -> str:
        if not self.artifact_root:
            return key  # fallback
        return os.path.join(self.artifact_root, f"{key}.pkl")