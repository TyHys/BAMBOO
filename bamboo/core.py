from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Type

import pandas as pd
from pydantic import BaseModel, create_model
from tqdm import tqdm

from .client import LLMClient
from .models import model_to_openai_schema, parse_model
from .cache import SimpleCache, NullCache


logger = logging.getLogger(__name__)


class LLMDataFrame:
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        client: Optional[LLMClient] = None,
        cache_path: Optional[str] = None,
        use_cache: bool = True,
    ) -> None:
        self.df = df
        self.client = client or LLMClient()
        if use_cache:
            self.cache = SimpleCache(cache_path)
        else:
            self.cache = NullCache()

    def enrich(
        self,
        *,
        input_col: str,
        response_model: Type[BaseModel],
        prompt_template: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        progress: bool = True,
    ) -> pd.DataFrame:
        """Enrich DataFrame by applying LLM with structured response.

        Adds new columns for each field in the response_model.
        Returns updated DataFrame.
        """
        if input_col not in self.df.columns:
            raise KeyError(f"Input column '{input_col}' not in DataFrame")

        schema = model_to_openai_schema(response_model)
        outputs: List[BaseModel] = []
        usages: List[Dict[str, Any]] = []

        iterator: Iterable[Any] = self.df[input_col]
        if progress:
            iterator = tqdm(iterator, total=len(self.df), desc="LLM enrich")

        for value in iterator:
            user_prompt = prompt_template.format(value=value)
            cache_key = f"{prompt_template}\n\n{value}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                content = cached
                usage = None
            else:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})
                content, usage = self.client.chat_structured(
                    messages,
                    response_schema=schema,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self.cache.set(cache_key, content)

            try:
                parsed = parse_model(response_model, content)
            except Exception as e:
                logger.warning("Row parse failed; inserting None values. Error: %s", e)
                parsed = None  # type: ignore[assignment]

            outputs.append(parsed)
            if usage:
                usages.append(usage)

        # Add columns
        if outputs:
            first_valid = next((o for o in outputs if o is not None), None)
            if first_valid is not None:
                for field_name in first_valid.model_fields.keys():
                    self.df[field_name] = [getattr(o, field_name, None) if o is not None else None for o in outputs]

        # Basic usage logging
        if usages:
            total = sum(u.get("total_tokens") or 0 for u in usages)
            logger.info("Processed %d rows, total_tokens=%s", len(outputs), total)

        return self.df

    # Stretch goal placeholder for future batching API
    def batch_enrich(self, *args: Any, **kwargs: Any) -> pd.DataFrame:  # pragma: no cover
        """Batch version of enrich() that processes rows in chunks with one LLM call per chunk.

        Accepts the same parameters as enrich() plus:
        - batch_size: number of rows per request (default 5)
        """
        input_col: str = kwargs.pop("input_col")
        response_model: Type[BaseModel] = kwargs.pop("response_model")
        prompt_template: str = kwargs.pop("prompt_template")
        system_prompt: Optional[str] = kwargs.pop("system_prompt", None)
        model: Optional[str] = kwargs.pop("model", None)
        temperature: float = kwargs.pop("temperature", 0.0)
        max_tokens: Optional[int] = kwargs.pop("max_tokens", None)
        progress: bool = kwargs.pop("progress", True)
        batch_size: int = kwargs.pop("batch_size", 5)

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

        if input_col not in self.df.columns:
            raise KeyError(f"Input column '{input_col}' not in DataFrame")

        # Build a wrapper model: { results: List[response_model] }
        BatchModel = create_model(
            "BatchResponse",
            results=(List[response_model], ...),
        )

        schema = model_to_openai_schema(BatchModel)

        values: List[Any] = list(self.df[input_col])
        outputs: List[Optional[BaseModel]] = []
        usages: List[Dict[str, Any]] = []

        def chunk_iter(seq: List[Any], size: int) -> Iterable[List[Any]]:
            for i in range(0, len(seq), size):
                yield seq[i : i + size]

        iterator: Iterable[List[Any]] = list(chunk_iter(values, batch_size))
        if progress:
            iterator = tqdm(iterator, total=(len(values) + batch_size - 1) // batch_size, desc="LLM batch enrich")

        for chunk in iterator:
            # Construct a batch prompt
            items_lines = []
            for idx, v in enumerate(chunk, start=1):
                items_lines.append(f"{idx}) {v}")
            items_block = "\n".join(items_lines)
            batch_instruction = (
                "Analyze the following items. For each item, produce an object matching the schema. "
                "Return a JSON object with key 'results' as a list of objects in the SAME order as inputs.\n"
                f"Items:\n{items_block}"
            )

            messages: List[Dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            # Include the single-item prompt_template to orient the model, then add the batch payload
            messages.append({"role": "user", "content": prompt_template.replace("{value}", "<item>")})
            messages.append({"role": "user", "content": batch_instruction})

            content, usage = self.client.chat_structured(
                messages,
                response_schema=schema,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if usage:
                usages.append(usage)

            try:
                parsed_batch = parse_model(BatchModel, content)
                batch_results = getattr(parsed_batch, "results", [])
                # Extend outputs with parsed results (may be fewer if model errs)
                for item in batch_results:
                    outputs.append(item)
                # If fewer results returned than inputs, pad with None
                if len(batch_results) < len(chunk):
                    outputs.extend([None] * (len(chunk) - len(batch_results)))
            except Exception as e:
                logger.warning("Batch parse failed; inserting None values. Error: %s", e)
                outputs.extend([None] * len(chunk))

        # Add columns from outputs
        if outputs:
            first_valid = next((o for o in outputs if o is not None), None)
            if first_valid is not None:
                for field_name in first_valid.model_fields.keys():
                    self.df[field_name] = [getattr(o, field_name, None) if o is not None else None for o in outputs]

        if usages:
            total = sum(u.get("total_tokens") or 0 for u in usages)
            logger.info("Processed %d rows (batched), total_tokens=%s", len(outputs), total)

        return self.df
