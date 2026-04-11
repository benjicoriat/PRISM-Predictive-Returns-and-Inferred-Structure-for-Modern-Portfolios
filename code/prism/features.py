from typing import Optional, Tuple

import numpy as np
import pandas as pd

_TARGET_CANDIDATES = (
    "target",
    "return",
    "returns",
    "future_return",
    "next_return",
    "y",
)


def infer_target_column(df: pd.DataFrame, explicit_target: Optional[str] = None) -> str:
    if explicit_target is not None:
        if explicit_target not in df.columns:
            raise ValueError(f"Target column '{explicit_target}' not found in dataset")
        return explicit_target

    lowered_map = {column.lower(): column for column in df.columns}
    for candidate in _TARGET_CANDIDATES:
        if candidate in lowered_map:
            return lowered_map[candidate]

    numeric_columns = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    if not numeric_columns:
        raise ValueError("Could not infer target column: no numeric columns found")

    return numeric_columns[-1]


def build_feature_matrix(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not present in dataframe")

    y = pd.to_numeric(df[target_column], errors="coerce")
    x_df = df.drop(columns=[target_column]).copy()

    if x_df.empty:
        raise ValueError("No feature columns remain after dropping target")

    x_df = pd.get_dummies(x_df, drop_first=True)
    x_df = x_df.apply(pd.to_numeric, errors="coerce")
    x_df = x_df.replace([np.inf, -np.inf], np.nan)

    x_df = x_df.fillna(x_df.median(numeric_only=True))
    x_df = x_df.fillna(0.0)

    valid_rows = y.notna()
    x_df = x_df.loc[valid_rows]
    y = y.loc[valid_rows]

    if x_df.shape[0] < 20:
        raise ValueError("Not enough valid rows to train a model (need at least 20)")

    return x_df, y
