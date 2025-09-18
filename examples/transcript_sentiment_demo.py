from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from bamboo import LLMDataFrame


class SentimentResult(BaseModel):
    score: int = Field(ge=0, le=100)
    explanation: str


def _choose_input_column(df: pd.DataFrame) -> str:
    if "Transcript" in df.columns:
        return "Transcript"
    if "Feedback" in df.columns:
        return "Feedback"
    raise KeyError("Expected either 'Transcript' or 'Feedback' column in the CSV.")


def main() -> None:
    input_path = Path("inputs/sample_data.csv")
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    input_col = _choose_input_column(df)

    system_prompt = (
        "You are ChatGPT. Return structured JSON only. "
        "For each input, assess sentiment and produce an integer 'score' from 0 to 100 "
        "(0=very negative, 50=neutral, 100=very positive) and a concise 'explanation'."
    )

    prompt_template = (
        "Analyze the following text and return a JSON object with fields 'score' and 'explanation'.\n"
        "Text: {value}"
    )

    llm_df = LLMDataFrame(df)
    df_out = llm_df.batch_enrich(
        input_col=input_col,
        response_model=SentimentResult,
        prompt_template=prompt_template,
        system_prompt=system_prompt,
        temperature=0.0,
        progress=True,
        batch_size=5,
    )

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "transcript_sentiment_scored.csv"
    df_out.to_csv(output_path, index=False)
    print(f"Wrote: {output_path}")
    print(df_out.head())


if __name__ == "__main__":
    main()

