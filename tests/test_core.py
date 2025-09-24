from __future__ import annotations

from typing import List

import pandas as pd
from pydantic import BaseModel

from bamboo.core import LLMDataFrame


class DummyModel(BaseModel):
    a: int
    b: str


class DummyClient:
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.calls = 0
        self.model = "test-model"

    def chat_structured(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        content = self.responses[self.calls]
        self.calls += 1
        return content, {"total_tokens": 10}


def test_enrich_adds_columns():
    df = pd.DataFrame({"text": ["x", "y"]})
    client = DummyClient(["{\"a\":1,\"b\":\"u\"}", "{\"a\":2,\"b\":\"v\"}"])
    llm_df = LLMDataFrame(df, client=client, use_cache=False)
    out = llm_df.enrich(
        response_model=DummyModel,
        user_prompt_template="Give me JSON for {text}",
        progress=False,
    )

    assert list(out.columns) == ["text", "a", "b"]
    assert out.loc[0, "a"] == 1
    assert out.loc[1, "b"] == "v"

