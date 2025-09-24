from __future__ import annotations

import os
import tempfile

from bamboo.cache import SimpleCache, NullCache


def test_simple_cache_get_set_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, ".bamboo_cache.json")
        c = SimpleCache(path)
        key = "hello"
        c.set(key, "world")
        assert c.get(key) == "world"

        # Reopen and ensure persistence
        c2 = SimpleCache(path)
        assert c2.get(key) == "world"


def test_null_cache_noop() -> None:
    n = NullCache()
    assert n.get("k") is None
    n.set("k", "v")
    assert n.get("k") is None


