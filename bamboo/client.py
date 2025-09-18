from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    from openai import OpenAI
    from openai import APIConnectionError, APIError, RateLimitError, BadRequestError, Timeout
except Exception:  # pragma: no cover - allow import in environments without openai
    OpenAI = None  # type: ignore
    APIConnectionError = APIError = RateLimitError = BadRequestError = Timeout = Exception  # type: ignore


class LLMClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini") -> None:
        # Try to load from a .env file if present
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv()
        except Exception:
            pass

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if OpenAI is None:
            raise RuntimeError("openai package not available. Please install 'openai>=1.0.0'.")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        self._client = OpenAI(api_key=api_key)
        self.model = model

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout, APIError)),
    )
    def chat_structured(
        self,
        messages: List[Dict[str, str]],
        *,
        response_schema: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """Call Chat Completions with structured JSON response enforcement.

        Returns (content, usage) where usage may include token counts.
        """
        chosen_model = model or self.model
        response_format: Dict[str, Any]
        if response_schema:
            response_format = {"type": "json_schema", "json_schema": response_schema}
        else:
            response_format = {"type": "json_object"}

        resp = self._client.chat.completions.create(
            model=chosen_model,
            messages=messages,
            temperature=temperature,
            response_format=response_format,  # Enforce JSON
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content or "{}"
        usage = None
        try:
            # openai 1.x returns usage with prompt_tokens and completion_tokens
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }
        except Exception:
            pass
        return content, usage
