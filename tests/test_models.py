from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from bamboo.models import model_to_openai_schema, extract_json_object, parse_model


class Child(BaseModel):
    x: int
    y: str


class Parent(BaseModel):
    name: str
    child: Child
    tags: List[str] = Field(default_factory=list)


def test_model_to_openai_schema_enforces_no_additional_properties() -> None:
    schema = model_to_openai_schema(Parent)
    assert schema["name"] == "Parent"
    assert schema["strict"] is True
    root = schema["schema"]
    # Root is object and disallows additional props
    assert root["type"] == "object"
    assert root["additionalProperties"] is False
    # Child is a $ref to $defs; resolve and verify
    child = root["properties"]["child"]
    assert "$ref" in child
    ref = child["$ref"]
    assert ref.startswith("#/$defs/")
    def_name = ref.split("/")[-1]
    child_def = root["$defs"][def_name]
    assert child_def["type"] == "object"
    assert child_def["additionalProperties"] is False


def test_extract_json_object_variants() -> None:
    payload = '{"a": 1}'
    assert extract_json_object(payload) == payload

    fenced = """
Here is output
```json
{"a": 1, "b": "c"}
```
Thanks
""".strip()
    out = extract_json_object(fenced)
    assert out.strip().startswith("{") and out.strip().endswith("}")


class Result(BaseModel):
    a: int
    b: str


def test_parse_model_success_and_failure() -> None:
    ok = '{"a": 1, "b": "x"}'
    model = parse_model(Result, ok)
    assert model.a == 1 and model.b == "x"

    bad = '{"a": 1, "b": }'
    try:
        parse_model(Result, bad)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # ValidationError path: JSON parses but fails model validation
    invalid = '{"a": 1}'  # missing required field b
    try:
        parse_model(Result, invalid)
        assert False, "Expected ValidationError"
    except Exception as e:
        # Pydantic raises ValidationError; our function re-raises it
        from pydantic import ValidationError  # local import for test
        assert isinstance(e, ValidationError)


