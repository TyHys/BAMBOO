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
    
    df = pd.DataFrame({
        "text": [
            "I love this product because it always brightens my day instantly.",
            "This is genuinely impressive and exceeded my expectations in every possible way.",
            "It’s okay, but honestly I discovered more good than I first expected.",
            "I would recommend this to everyone since it brings consistent happiness daily.",
            "Using this has been such a joy, it feels wonderful every time.",
            "The overall experience was excellent and I cannot wait to enjoy again.",
            "I’m delighted with how well this turned out, simply a great choice.",
            "Every time I use this I feel better, happier, and more energized.",
            "This has been one of the most enjoyable experiences I have ever had.",
            "I’m very pleased with this and look forward to recommending it widely."
        ],
        "category": [
            "consumer electronics",
            "home appliances",
            "books",
            "fitness",
            "travel",
            "software",
            "food & beverage",
            "health & wellness",
            "entertainment",
            "fashion"
        ]
    })

    llm_df = LLMDataFrame(df)

    df2 = llm_df.enrich(
        input_cols=["text", "category"],
        response_model=SentimentOutput,
        prompt_template=(
            "Analyze the following product feedback within its category and return JSON matching the schema.\n"
            "Category: {category}\n"
            "Text: {text}"
        ),
        model="gpt-4o-mini",
    )

    print(df2)


if __name__ == "__main__":
    main()

