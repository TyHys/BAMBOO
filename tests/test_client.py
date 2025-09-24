from __future__ import annotations

import os
import importlib
import sys

import pytest


def test_client_import_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate openai import failure and ensure RuntimeError on construction
    monkeypatch.setitem(sys.modules, 'openai', None)
    from bamboo import client as client_mod  # import after monkeypatch
    importlib.reload(client_mod)

    from bamboo.client import LLMClient
    with pytest.raises(RuntimeError):
        LLMClient(api_key=os.environ.get("OPENAI_API_KEY"))


