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


def test_enrich_batched_deduplicates_unique_contexts() -> None:
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

    out = df.bamboo.enrich(
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


def test_resume_path_reuses_results(tmp_path: Any) -> None:
    df = pd.DataFrame({"text": ["a", "b", "a"]})

    class M(BaseModel):
        a: int
        b: str

    # First run returns two unique results (for 'a' and 'b')
    first_contents = [
        "{\"results\":[{\"a\":1,\"b\":\"x\"},{\"a\":2,\"b\":\"y\"}]}",
    ]
    client1 = DummyClient(first_contents)
    resume_file = tmp_path / "resume.json"
    out1 = df.bamboo.enrich(
        client=client1,
        response_model=M,
        user_prompt_template="Return a and b for: {text}",
        system_prompt_template=None,
        batch_size=10,
        progress=False,
        use_cache=False,
        resume_path=str(resume_file),
    )
    assert {"a", "b"}.issubset(out1.columns)
    assert client1.calls == 1

    # Second run with a client that would error if called: ensure no new calls
    client2 = DummyClient(contents=[])
    out2 = df.bamboo.enrich(
        client=client2,
        response_model=M,
        user_prompt_template="Return a and b for: {text}",
        system_prompt_template=None,
        batch_size=10,
        progress=False,
        use_cache=False,
        resume_path=str(resume_file),
    )
    assert client2.calls == 0
    # Data should match the first run
    assert out2.equals(out1)


