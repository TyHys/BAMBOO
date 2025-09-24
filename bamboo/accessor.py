from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from pandas.api.extensions import register_dataframe_accessor

from .core import LLMDataFrame
from .client import LLMClient


@register_dataframe_accessor("bamboo")
class BambooAccessor:
    """Pandas accessor to call BAMBOO directly on any DataFrame.

    Example:
        df_out = df.bamboo.enrich(response_model=..., user_prompt_template=..., system_prompt_template=...)
    """

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        self._df = pandas_obj

    def enrich(
        self,
        *,
        client: Optional[LLMClient] = None,
        cache_path: Optional[str] = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        llm_df = LLMDataFrame(self._df, client=client, cache_path=cache_path, use_cache=use_cache)
        return llm_df.enrich(**kwargs)


