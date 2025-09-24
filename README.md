# BAMBOO

<p align="center">
  <img src="logo.png" alt="BAMBOO Logo" />
</p>


Boosted Augmentation for Machine-Based Output on Objects  🐼

BAMBOO enriches your pandas DataFrames using LLMs and returns structured outputs via Pydantic models. It supports both row-by-row processing and efficient batch processing (via `batch_size`) to reduce API calls and latency.

## Features
- Structured LLM outputs validated by Pydantic
- Integrated API: `df.bamboo.enrich(..., batch_size=...)`
- Batch mode: send multiple rows in a single LLM request by setting `batch_size>1`
- Automatic input inference from template placeholders (no need for `input_col` if templates reference DataFrame columns)
- Unique-combination deduplication: only one LLM call per unique combination of prompt-included fields, with results broadcast back to matching rows.
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

df = pd.DataFrame({"text": ["I love this!", "This is bad.", "It’s okay."]})

class SentimentResult(BaseModel):
    score: int = Field(ge=0, le=100, description="The score of the review, with 0 being the most negative and 100 being the most positive.")
    explanation: str = Field(description="The explanation of the score.")

system_prompt_template = "You are an assistant that analyzes reviews score and explanation."

user_prompt_template = "Analyze the review and return 'score' and 'explanation' for: {text}"

df_out = df.bamboo.enrich(
    response_model=SentimentResult,
    user_prompt_template=user_prompt_template,
    system_prompt_template=system_prompt_template,
    temperature=0.0,
    batch_size=5
)

print(df_out)
```

### Recommended usage
- Use the pandas accessor method: `df.bamboo.enrich(..., batch_size=...)`.

### Progress bar
- The progress bar (tqdm) is disabled by default. Enable it per call with `progress=True`:
  ```python
  df.bamboo.enrich(..., progress=True)
  ```

### Multi-column templating
Reference multiple columns by name directly in your templates.
```python
df = pd.DataFrame({
    "text": ["Great build quality."],
    "category": ["consumer electronics"],
})

# Placeholders match DataFrame columns, so inputs are inferred automatically
out = df.bamboo.enrich(
    response_model=SentimentResult,
    user_prompt_template=(
        "Analyze the following product feedback within its category and return JSON.\n"
        "Category: {category}\n"
        "Text: {text}"
    ),
)
```

### Automatic input inference (no input_cols)
- Inputs are inferred from placeholders found in any of: `user_prompt_template` or `system_prompt_template`.
- If no placeholders are found, an error is raised; add `{col_name}` placeholders to your templates.
- If any placeholder does not match a DataFrame column, an error lists the missing columns.

### Unique-combination optimization (automatic)
- BAMBOO detects which fields in your templates actually affect the prompt and deduplicates rows by the unique combinations of those fields. It runs the LLM once per unique combination and broadcasts the parsed result back to all matching rows.
- Works in both row-by-row and batched modes transparently via `enrich(..., batch_size=...)`.
- Example: your template references only `{department}` and `{severity}`; for 1,000 rows with 8 unique `(department, severity)` pairs, only 8 LLM inferences are made.

## Demos (Notebooks)
- Sentiment Analysis (notebook): `examples/Sentiment Analysis.ipynb`
- Ticket Triage (notebook): `examples/Ticket Triage.ipynb`

Open these in Jupyter or VS Code and run the cells. To ensure the notebook uses your local BAMBOO checkout, install it in editable mode from the repo root:
```bash
%pip install -e .
```
If running from a subdirectory, you can use:
```bash
%pip install -e "$(git rev-parse --show-toplevel)"
```
After install, restart the kernel to pick up changes:
```python
from IPython import get_ipython
get_ipython().kernel.do_shutdown(True)
```

## Caching
- Cache file: `.bamboo_cache.json` in the working directory
- Toggle per call with the pandas accessor (recommended):
  ```python
  # Disable cache for this call
  df.bamboo.enrich(
      response_model=SentimentResult,
      user_prompt_template=user_prompt_template,
      system_prompt_template=system_prompt_template,
      use_cache=False,
  )

  # Custom cache path
  df.bamboo.enrich(
      response_model=SentimentResult,
      user_prompt_template=user_prompt_template,
      system_prompt_template=system_prompt_template,
      cache_path="./my_cache/bamboo_cache.json",
  )
  ```

## Models and Strict JSON
BAMBOO enforces strict JSON with OpenAI’s `response_format=json_schema`. The Pydantic model’s JSON schema is adapted to set `additionalProperties=false` across all object nodes for consistent parsing.

## Requirements
- Python 3.10+
- OpenAI account and API key


- Linting: this repo relies on type hints and basic style; integrate your preferred linter if desired.

## License
MIT

## To-do

* Experiment with various models to ensure compatability.
* Realtime file writing
* Resume enrichment from partial completion
* Add Claude support
* Add Gemini support
