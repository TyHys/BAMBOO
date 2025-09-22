from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Type, Union

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

    @staticmethod
    def _safe_format(template: str, context: Dict[str, Any]) -> str:
        """Best-effort .format_map that leaves unknown placeholders intact.

        Example: "Hello {name} {missing}" with {"name":"A"} -> "Hello A {missing}"
        """
        class _SafeDict(dict):  # type: ignore[type-arg]
            def __missing__(self, key: str) -> str:  # pragma: no cover - trivial
                return "{" + key + "}"

        return template.format_map(_SafeDict(context))

    @staticmethod
    def _schema_hash(schema: Dict[str, Any]) -> str:
        try:
            dumped = json.dumps(schema, sort_keys=True, ensure_ascii=False)
        except Exception:
            dumped = str(schema)
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

    def enrich(
        self,
        *,
        input_col: Optional[str] = None,
        response_model: Type[BaseModel],
        prompt_template: str,
        system_prompt: Optional[str] = None,
        input_cols: Optional[Union[List[str], Dict[str, str]]] = None,
        system_prompt_template: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        progress: bool = True,
    ) -> pd.DataFrame:
        """Enrich DataFrame by applying LLM with structured response.

        Adds new columns for each field in the response_model.
        Returns updated DataFrame.
        """
        if input_cols is None:
            if not input_col or input_col not in self.df.columns:
                raise KeyError(f"Input column '{input_col}' not in DataFrame")
        else:
            # Validate provided columns if list; if dict, validate mapped DataFrame columns
            if isinstance(input_cols, list):
                missing = [c for c in input_cols if c not in self.df.columns]
                if missing:
                    raise KeyError(f"Input columns missing from DataFrame: {missing}")
            else:
                missing = [src for src in input_cols.values() if src not in self.df.columns]
                if missing:
                    raise KeyError(f"Input columns missing from DataFrame: {missing}")

        schema = model_to_openai_schema(response_model)
        outputs: List[BaseModel] = []
        usages: List[Dict[str, Any]] = []

        # Build an iterator of contexts or plain values based on input_cols
        if input_cols is None:
            iterator: Iterable[Any] = self.df[input_col]
        else:
            contexts: List[Dict[str, Any]] = []
            if isinstance(input_cols, list):
                for _, row in self.df.iterrows():
                    ctx: Dict[str, Any] = {name: row[name] for name in input_cols}
                    contexts.append(ctx)
            else:
                # Mapping of template_name -> dataframe_column
                for _, row in self.df.iterrows():
                    ctx = {tpl_name: row[src_col] for tpl_name, src_col in input_cols.items()}
                    contexts.append(ctx)
            iterator = contexts

        chosen_model = model or self.client.model
        schema_h = self._schema_hash(schema)
        if progress:
            iterator = tqdm(iterator, total=len(self.df), desc="LLM enrich")

        for item in iterator:
            if input_cols is None:
                # Single column mode
                user_prompt = prompt_template.format(value=item)
                final_system_prompt = system_prompt
                cache_context_repr = json.dumps({"value": item}, ensure_ascii=False, default=str)
            else:
                # Multi-column templating mode
                context: Dict[str, Any] = item  # type: ignore[assignment]
                user_prompt = self._safe_format(prompt_template, context)
                if system_prompt_template is not None:
                    final_system_prompt = self._safe_format(system_prompt_template, context)
                else:
                    final_system_prompt = system_prompt
                cache_context_repr = json.dumps(context, ensure_ascii=False, default=str, sort_keys=True)

            cache_key = "\n".join(
                [
                    "bamboo.enrich",
                    f"model={chosen_model}",
                    f"temperature={temperature}",
                    f"max_tokens={max_tokens}",
                    f"schema={schema_h}",
                    f"prompt_template={prompt_template}",
                    f"system_prompt={system_prompt or ''}",
                    f"system_prompt_template={system_prompt_template or ''}",
                    f"context={cache_context_repr}",
                ]
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                content = cached
                usage = None
            else:
                messages = []
                if final_system_prompt:
                    messages.append({"role": "system", "content": final_system_prompt})
                messages.append({"role": "user", "content": user_prompt})
                content, usage = self.client.chat_structured(
                    messages,
                    response_schema=schema,
                    model=chosen_model,
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
        input_col: Optional[str] = kwargs.pop("input_col", None)
        response_model: Type[BaseModel] = kwargs.pop("response_model")
        prompt_template: str = kwargs.pop("prompt_template")
        system_prompt: Optional[str] = kwargs.pop("system_prompt", None)
        input_cols: Optional[Union[List[str], Dict[str, str]]] = kwargs.pop("input_cols", None)
        system_prompt_template: Optional[str] = kwargs.pop("system_prompt_template", None)
        model: Optional[str] = kwargs.pop("model", None)
        temperature: float = kwargs.pop("temperature", 0.0)
        max_tokens: Optional[int] = kwargs.pop("max_tokens", None)
        progress: bool = kwargs.pop("progress", True)
        batch_size: int = kwargs.pop("batch_size", 5)

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

        if input_cols is None:
            if not input_col or input_col not in self.df.columns:
                raise KeyError(f"Input column '{input_col}' not in DataFrame")
        else:
            if isinstance(input_cols, list):
                missing = [c for c in input_cols if c not in self.df.columns]
                if missing:
                    raise KeyError(f"Input columns missing from DataFrame: {missing}")
            else:
                missing = [src for src in input_cols.values() if src not in self.df.columns]
                if missing:
                    raise KeyError(f"Input columns missing from DataFrame: {missing}")

        # Build a wrapper model: { results: List[response_model] }
        BatchModel = create_model(
            "BatchResponse",
            results=(List[response_model], ...),
        )

        schema = model_to_openai_schema(BatchModel)

        if input_cols is None:
            values: List[Any] = list(self.df[input_col])
        else:
            values = []
            if isinstance(input_cols, list):
                for _, row in self.df.iterrows():
                    values.append({name: row[name] for name in input_cols})
            else:
                for _, row in self.df.iterrows():
                    values.append({tpl: row[src] for tpl, src in input_cols.items()})
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
            items_lines: List[str] = []
            if input_cols is None:
                for idx, v in enumerate(chunk, start=1):
                    items_lines.append(f"{idx}) {v}")
            else:
                for idx, ctx in enumerate(chunk, start=1):
                    ctx_json = json.dumps(ctx, ensure_ascii=False, default=str, sort_keys=True)
                    items_lines.append(f"{idx}) {ctx_json}")
            items_block = "\n".join(items_lines)
            batch_instruction = (
                "Analyze the following items. For each item, produce an object matching the schema. "
                "Return a JSON object with key 'results' as a list of objects in the SAME order as inputs.\n"
                f"Items (each item is {'a raw value' if input_cols is None else 'a JSON object of fields'}):\n{items_block}"
            )

            messages: List[Dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            # Include the single-item prompt_template to orient the model (static part only), then add the batch payload
            messages.append({"role": "user", "content": self._safe_format(prompt_template, {})})
            if system_prompt_template:
                # Not per-item in batch mode; include as a generic note if provided
                messages.append({"role": "user", "content": self._safe_format(system_prompt_template, {})})
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
