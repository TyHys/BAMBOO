from __future__ import annotations

import json
import re
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError


ModelT = TypeVar("ModelT", bound=BaseModel)


def model_to_openai_schema(model_cls: Type[ModelT]) -> dict[str, Any]:
    """Convert a Pydantic model to an OpenAI json_schema response format.

    Returns a dict suitable for response_format={"type":"json_schema","json_schema":{...}}
    """
    schema = model_cls.model_json_schema()
    if not isinstance(schema, dict):
        raise TypeError("model_json_schema() did not return a dict")

    # Ensure OpenAI response_format requirements on all object nodes
    def enforce_no_additional_props(node: Any) -> None:
        if isinstance(node, dict):
            node_type = node.get("type")
            # If this is an object (explicitly or by presence of properties), disallow extras
            if node_type == "object" or ("properties" in node and node_type is None):
                node.setdefault("type", "object")
                node["additionalProperties"] = False
            # Recurse through typical schema containers
            for key in ("properties", "$defs", "definitions"):
                if key in node and isinstance(node[key], dict):
                    for _, child in node[key].items():
                        enforce_no_additional_props(child)
            # Recurse into items for arrays
            if "items" in node:
                enforce_no_additional_props(node["items"])
        elif isinstance(node, list):
            for item in node:
                enforce_no_additional_props(item)

    enforce_no_additional_props(schema)

    # OpenAI requires a name for the schema root
    name = getattr(model_cls, "__name__", "Response")
    return {
        "name": name,
        "schema": schema,
        "strict": True,
    }


def extract_json_object(text: str) -> str:
    """Best-effort extraction of a JSON object from a text blob."""
    # Quick path
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # Fallback: greedy extract from first '{' to last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        # Remove code fences if present
        candidate = re.sub(r"^```(json)?\n|```$", "", candidate.strip(), flags=re.IGNORECASE | re.MULTILINE)
        return candidate
    return text


def parse_model(model_cls: Type[ModelT], text: str) -> ModelT:
    payload = extract_json_object(text)
    try:
        data = json.loads(payload)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {e}") from e
    try:
        return model_cls.model_validate(data)
    except ValidationError as e:
        raise
