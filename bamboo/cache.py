import hashlib
import json
import os
import threading
from typing import Any, Optional


class SimpleCache:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or os.path.join(os.getcwd(), ".bamboo_cache.json")
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def _save(self) -> None:
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)

    @staticmethod
    def _hash_key(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[str]:
        hashed = self._hash_key(key)
        with self._lock:
            return self._data.get(hashed)

    def set(self, key: str, value: str) -> None:
        hashed = self._hash_key(key)
        with self._lock:
            self._data[hashed] = value
            self._save()


class NullCache(SimpleCache):
    def __init__(self) -> None:  # type: ignore[no-untyped-def]
        pass

    def get(self, key: str) -> Optional[str]:  # type: ignore[override]
        return None

    def set(self, key: str, value: str) -> None:  # type: ignore[override]
        return None
