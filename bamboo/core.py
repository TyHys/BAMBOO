from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Type, Union, Set
from string import Formatter

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
    def _extract_placeholders(template: Optional[str]) -> Set[str]:
        """Extract placeholder field names from a format template string.

        Returns a set of field names found inside curly braces, ignoring None templates.
        """
        if not template:
            return set()
        fields: Set[str] = set()
        for literal_text, field_name, format_spec, conversion in Formatter().parse(template):  # type: ignore[assignment]
            if field_name:
                # Handle attribute/index lookups like {user[name]} by taking root before any punctuation
                root = field_name.split(".")[0].split("[")[0]
                if root:
                    fields.add(root)
        return fields

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
        user_prompt_template: Optional[str] = None,
        input_cols: Optional[Union[List[str], Dict[str, str]]] = None,
        system_prompt_template: Optional[str] = None,
        # Backward-compat: accept legacy name; prefer user_prompt_template when both provided
        prompt_template: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        progress: bool = False,
        batch_size: int = 1,
        resume_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Enrich DataFrame by applying LLM with structured response.

        Adds new columns for each field in the response_model.
        Returns updated DataFrame.
        """
        # Normalize prompt parameter
        if user_prompt_template is None and prompt_template is not None:
            user_prompt_template = prompt_template
        if user_prompt_template is None:
            raise TypeError("user_prompt_template is required")

        # input_cols is obsolete
        if input_cols is not None:
            raise TypeError("input_cols is obsolete. Reference DataFrame columns in prompt_template/system_prompt_template instead.")

        # Determine required fields from templates (user_prompt_template, system_prompt_template)
        inferred_fields = (
            self._extract_placeholders(user_prompt_template)
            | self._extract_placeholders(system_prompt_template)
        )
        if not inferred_fields:
            raise KeyError(
                "No input columns inferred from templates. Add placeholders (e.g., {col_name}) to prompt_template or system_prompt_template."
            )
        missing = [f for f in inferred_fields if f not in self.df.columns]
        if missing:
            raise KeyError(f"Template placeholders missing from DataFrame: {missing}")

        schema = model_to_openai_schema(response_model)
        outputs: List[BaseModel] = []
        usages: List[Dict[str, Any]] = []

        # Build contexts in-memory from inferred fields (needed for deduplication)
        contexts: List[Dict[str, Any]] = []
        for _, row in self.df.iterrows():
            ctx = {name: row[name] for name in inferred_fields}
            contexts.append(ctx)

        # Determine which fields are actually used by the templates
        used_fields = (
            self._extract_placeholders(user_prompt_template)
            | self._extract_placeholders(system_prompt_template)
        ) & set(inferred_fields)

        # Compute signatures for deduplication based on used fields only
        def signature_for_context(ctx: Dict[str, Any]) -> str:
            if not used_fields:
                # No fields used -> static prompt
                return "{}"
            used_subset = {k: ctx.get(k) for k in sorted(used_fields)}
            return json.dumps(used_subset, ensure_ascii=False, default=str, sort_keys=True)

        signatures: List[str] = [signature_for_context(c) for c in contexts]

        # Map unique signature -> representative index
        sig_to_rep_index: Dict[str, int] = {}
        rep_indices: List[int] = []
        for idx, sig in enumerate(signatures):
            if sig not in sig_to_rep_index:
                sig_to_rep_index[sig] = idx
                rep_indices.append(idx)

        chosen_model = model or self.client.model
        schema_h = self._schema_hash(schema)
        resume_results: Dict[str, Optional[str]] = {}

        # Resume support: load existing results map if provided and compatible
        def _resume_meta_dict() -> Dict[str, Any]:
            return {
                "version": 1,
                "model": chosen_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "schema_hash": schema_h,
                "user_prompt_template": user_prompt_template,
                "system_prompt_template": system_prompt_template or "",
            }

        def _resume_load(path: str) -> None:
            nonlocal resume_results
            if not os.path.exists(path):
                return
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                return
            meta = data.get("meta")
            if meta != _resume_meta_dict():
                return
            loaded = data.get("results") or {}
            if isinstance(loaded, dict):
                # keep only string or None values
                filtered: Dict[str, Optional[str]] = {}
                for k, v in loaded.items():
                    if v is None or isinstance(v, str):
                        filtered[k] = v
                resume_results.update(filtered)

        def _resume_save(path: str) -> None:
            if not path:
                return
            data = {"meta": _resume_meta_dict(), "results": resume_results}
            tmp = path + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, path)
            except Exception:
                # Best effort; ignore persistence errors
                pass

        if resume_path:
            _resume_load(resume_path)

        # Hold results per signature and then broadcast
        result_by_signature: Dict[str, Optional[BaseModel]] = {}

        # Prefill from resume file if present
        if resume_results:
            for sig, content in resume_results.items():
                if content is None:
                    result_by_signature[sig] = None
                    continue
                try:
                    parsed = parse_model(response_model, content)
                except Exception:
                    parsed = None  # ignore parse errors from resume file
                result_by_signature[sig] = parsed

        if batch_size <= 1:
            rep_iterator: Iterable[int] = rep_indices
            if progress:
                rep_iterator = tqdm(rep_indices, total=len(rep_indices), desc="LLM enrich (unique)")

            for rep_idx in rep_iterator:
                context = contexts[rep_idx]
                sig = signatures[rep_idx]

                # Skip if already available (e.g., from resume file)
                if sig in result_by_signature:
                    continue

                # Build prompts
                user_prompt = self._safe_format(user_prompt_template, context)
                if system_prompt_template is not None:
                    final_system_prompt = self._safe_format(system_prompt_template, context)
                else:
                    final_system_prompt = None
                cache_context_repr = json.dumps(context, ensure_ascii=False, default=str, sort_keys=True)

                cache_key = "\n".join(
                    [
                        "bamboo.enrich",
                        f"model={chosen_model}",
                        f"temperature={temperature}",
                        f"max_tokens={max_tokens}",
                        f"schema={schema_h}",
                        f"user_prompt_template={user_prompt_template}",
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

                result_by_signature[sig] = parsed
                if resume_path:
                    resume_results[sig] = content if content is not None else None
                    _resume_save(resume_path)
                if usage:
                    usages.append(usage)
        else:
            # Batched mode: create a wrapper model and process unique contexts in chunks
            BatchModel = create_model(
                "BatchResponse",
                results=(List[response_model], ...),
            )
            batch_schema = model_to_openai_schema(BatchModel)

            # Build unique lists in representative order
            unique_contexts: List[Dict[str, Any]] = [contexts[i] for i in rep_indices]
            unique_signatures: List[str] = [signatures[i] for i in rep_indices]

            # Prepare iterator over chunks
            def chunk_iter(seq: List[Any], size: int) -> Iterable[List[Any]]:
                for i in range(0, len(seq), size):
                    yield seq[i : i + size]

            # Consider only remaining (not yet available) signatures
            remaining_indices: List[int] = [i for i, sig in enumerate(unique_signatures) if sig not in result_by_signature]
            remaining_contexts: List[Dict[str, Any]] = [unique_contexts[i] for i in remaining_indices]
            remaining_signatures: List[str] = [unique_signatures[i] for i in remaining_indices]

            chunks: List[List[int]] = [
                remaining_indices[i : i + batch_size]
                for i in range(0, len(remaining_indices), batch_size)
            ]
            iterator: Iterable[List[int]] = chunks
            if progress:
                iterator = tqdm(chunks, total=len(chunks), desc="LLM enrich (batched unique)")

            for index_chunk in iterator:
                if not index_chunk:
                    continue
                chunk_signatures = [unique_signatures[i] for i in index_chunk]
                chunk_contexts = [unique_contexts[i] for i in index_chunk]

                # Construct a batch prompt
                items_lines: List[str] = []
                for idx_local, ctx in enumerate(chunk_contexts, start=1):
                    ctx_json = json.dumps(ctx, ensure_ascii=False, default=str, sort_keys=True)
                    items_lines.append(f"{idx_local}) {ctx_json}")
                items_block = "\n".join(items_lines)
                batch_instruction = (
                    "Analyze the following items. For each item, produce an object matching the schema. "
                    "Return a JSON object with key 'results' as a list of objects in the SAME order as inputs.\n"
                    f"Items (each item is a JSON object of fields):\n{items_block}"
                )

                messages: List[Dict[str, str]] = []
                if system_prompt_template:
                    messages.append({"role": "system", "content": self._safe_format(system_prompt_template, {})})
                messages.append({"role": "user", "content": self._safe_format(user_prompt_template, {})})
                messages.append({"role": "user", "content": batch_instruction})

                content, usage = self.client.chat_structured(
                    messages,
                    response_schema=batch_schema,
                    model=chosen_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if usage:
                    usages.append(usage)

                try:
                    parsed_batch = parse_model(BatchModel, content)
                    batch_results = getattr(parsed_batch, "results", [])
                    for i in range(len(chunk_signatures)):
                        if i < len(batch_results):
                            result_by_signature[chunk_signatures[i]] = batch_results[i]
                            if resume_path:
                                # Store per-item JSON for resume: serialize the individual object
                                try:
                                    item_json = json.dumps(batch_results[i].model_dump(), ensure_ascii=False)
                                except Exception:
                                    item_json = None  # fall back to None if cannot serialize
                                resume_results[chunk_signatures[i]] = item_json
                        else:
                            result_by_signature[chunk_signatures[i]] = None
                            if resume_path:
                                resume_results[chunk_signatures[i]] = None
                except Exception as e:
                    logger.warning("Batch parse failed; inserting None values. Error: %s", e)
                    for sig in chunk_signatures:
                        result_by_signature[sig] = None
                        if resume_path:
                            resume_results[sig] = None

                if resume_path:
                    _resume_save(resume_path)

            # After batched processing, continue to broadcast below

        # Broadcast results to all rows
        outputs = [result_by_signature.get(sig) for sig in signatures]

        # Add columns
        if outputs:
            first_valid = next((o for o in outputs if o is not None), None)
            if first_valid is not None:
                for field_name in first_valid.__class__.model_fields.keys():
                    self.df[field_name] = [getattr(o, field_name, None) if o is not None else None for o in outputs]

        # Basic usage logging
        if usages:
            total = sum(u.get("total_tokens") or 0 for u in usages)
            if batch_size <= 1:
                logger.info("Processed %d rows, total_tokens=%s", len(outputs), total)
            else:
                logger.info(
                    "Processed %d rows (batched unique=%d), total_tokens=%s",
                    len(outputs),
                    len({s for s in signatures}),
                    total,
                )

        return self.df
