# BAMBOO

![BAMBOO Logo](logo.png)


Boosted Augmentation for Machine-Based Output on Objects  🐼

BAMBOO enriches your pandas DataFrames using LLMs and returns structured outputs via Pydantic models. It supports both row-by-row processing and efficient batch processing to reduce API calls and latency.

## Features
- Structured LLM outputs validated by Pydantic
- Preferred DataFrame API: `df.bamboo.enrich(...)` and `df.bamboo.batch_enrich(...)`
- Also available: `LLMDataFrame.enrich` and `LLMDataFrame.batch_enrich`
- Batch mode: send multiple rows in a single LLM request
- Automatic `.env` loading for `OPENAI_API_KEY`
- Lightweight on-disk caching to avoid re-calling unchanged prompts

## Install
```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Configure
Provide your OpenAI API key via either:
- `.env` file at repo root:
  ```env
  OPENAI_API_KEY=your_key
  ```
- or environment export:
  ```bash
  export OPENAI_API_KEY=your_key
  ```

## Quick Start (preferred accessor syntax)
```python
import pandas as pd
from pydantic import BaseModel, Field
import bamboo  # registers the pandas accessor: df.bamboo

class SentimentResult(BaseModel):
    score: int = Field(ge=0, le=100)
    explanation: str

df = pd.DataFrame({"text": ["I love this!", "This is bad.", "It’s okay."]})

system_prompt = (
    "Return JSON only. For each input, produce integer 'score' (0–100) and 'explanation'."
)
prompt_template = "Analyze and return {'score','explanation'} for: {value}"

# Row-by-row
out1 = df.bamboo.enrich(
    input_col="text",
    response_model=SentimentResult,
    prompt_template=prompt_template,
    system_prompt=system_prompt,
    temperature=0.0,
)

# Batched (recommended for larger DataFrames)
out2 = df.bamboo.batch_enrich(
    input_col="text",
    response_model=SentimentResult,
    prompt_template=prompt_template,
    system_prompt=system_prompt,
    temperature=0.0,
    batch_size=5,
)

print(out2[["text", "score", "explanation"]])
```

### Multi-column templating
You can provide multiple columns and reference them by name in the prompt template.
```python
df = pd.DataFrame({
    "text": ["Great build quality."],
    "category": ["consumer electronics"],
})

out = df.bamboo.enrich(
    input_cols=["text", "category"],
    response_model=SentimentResult,
    prompt_template=(
        "Analyze the following product feedback within its category and return JSON.\n"
        "Category: {category}\n"
        "Text: {text}"
    ),
)
```

## Demos
- Transcript/Feedback sentiment demo (writes `outputs/transcript_sentiment_scored.csv`):
```bash
python -m examples.transcript_sentiment_demo
```
- Minimal inline demo:
```bash
python -m examples.sentiment_demo
```

Input CSV expected at `inputs/sample_data.csv` with a `Transcript` column (fallback to `Feedback`).

## Caching
- Cache file: `.bamboo_cache.json` in the working directory
- Disable cache by constructing `LLMDataFrame(df, use_cache=False)` or pass a custom path via `cache_path`

## Models and Strict JSON
BAMBOO enforces strict JSON with OpenAI’s `response_format=json_schema`. The Pydantic model’s JSON schema is adapted to set `additionalProperties=false` across all object nodes for consistent parsing.

## Requirements
- Python 3.10+
- OpenAI account and API key

## Development
- Run tests (if any):
```bash
pytest -q
```
- Linting: this repo relies on type hints and basic style; integrate your preferred linter if desired.

## License
MIT
