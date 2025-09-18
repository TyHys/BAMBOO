from __future__ import annotations

from typing import List

import pandas as pd
from pydantic import BaseModel

from bamboo import LLMDataFrame


class SentimentOutput(BaseModel):
    sentiment: str
    category: str
    keywords: List[str]


def main() -> None:
    df = pd.DataFrame({"text": ["I love this!", "This is bad.", "It’s okay."]})
    llm_df = LLMDataFrame(df)
    df2 = llm_df.enrich(
        input_col="text",
        response_model=SentimentOutput,
        prompt_template=(
            "Analyze this text and return sentiment, category, and 3 keywords as JSON.\n"
            "Text: {value}"
        ),
    )
    print(df2)


if __name__ == "__main__":
    main()

