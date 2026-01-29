"""Utility filters for comparison outputs."""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _filter_self_comparisons(
    pairwise_df: pd.DataFrame, *, context: Optional[str] = None
) -> pd.DataFrame:
    """Drop self-comparisons where A and B are identical."""
    if pairwise_df is None or pairwise_df.empty:
        return pairwise_df
    if "A" not in pairwise_df.columns or "B" not in pairwise_df.columns:
        return pairwise_df

    normalized_a = pairwise_df["A"].astype(str).str.strip()
    normalized_b = pairwise_df["B"].astype(str).str.strip()
    same_mask = (
        pairwise_df["A"].notna()
        & pairwise_df["B"].notna()
        & (normalized_a == normalized_b)
    )
    if same_mask.any():
        logger.debug(
            "Dropping %d self-comparison rows%s",
            int(same_mask.sum()),
            f" for {context}" if context else "",
        )
        return pairwise_df.loc[~same_mask].copy()
    return pairwise_df
