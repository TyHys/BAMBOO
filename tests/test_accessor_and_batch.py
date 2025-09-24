from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel

import bamboo  # noqa: F401  # registers df.bamboo accessor


class DummyModel(BaseModel):
    a: int
    b: str


class DummyClient:
    def __init__(self, contents: List[str]) -> None:
        self.contents = contents
        self.calls = 0
        self.model = "dummy"

    def chat_structured(
        self,
        messages: List[Dict[str, str]],
        *,
        response_schema: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        content = self.contents[self.calls]
        self.calls += 1
        return content, {"total_tokens": 5}


def test_accessor_enrich_adds_columns() -> None:
    df = pd.DataFrame({"text": ["x", "y"]})
    client = DummyClient(["{\"a\":1,\"b\":\"u\"}", "{\"a\":2,\"b\":\"v\"}"])
    out = df.bamboo.enrich(
        client=client,
        response_model=DummyModel,
        user_prompt_template="Give me JSON for {text}",
        progress=False,
        use_cache=False,
    )
    assert list(out.columns) == ["text", "a", "b"]
    assert out.loc[0, "a"] == 1 and out.loc[1, "b"] == "v"


def test_batch_enrich_deduplicates_unique_contexts() -> None:
    rows = [
        {"department": "sales", "severity": "low"},
        {"department": "support", "severity": "high"},
        {"department": "sales", "severity": "low"},  # duplicate of row 0
    ]
    df = pd.DataFrame(rows)

    # One batch response for the two unique contexts
    batch_content = (
        "{"
        "\"results\":["
        "{\"a\":1,\"b\":\"ok\"},"
        "{\"a\":2,\"b\":\"alert\"}"
        "]}"
    )
    client = DummyClient([batch_content])

    out = df.bamboo.batch_enrich(
        client=client,
        response_model=DummyModel,
        user_prompt_template=(
            "Decide. Department: {department}\nSeverity: {severity}\nReturn fields a and b."
        ),
        progress=False,
        batch_size=10,
        use_cache=False,
    )

    # Columns present
    assert {"a", "b"}.issubset(out.columns)
    # Dedupe ensured a single client call for 2 unique contexts
    assert client.calls == 1
    # Broadcast results: first and third rows share results
    assert out.loc[0, "a"] == out.loc[2, "a"]


