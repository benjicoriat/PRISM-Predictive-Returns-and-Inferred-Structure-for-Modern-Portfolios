from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

SourceType = Literal["csv", "synthetic"]


def load_dataset(data_path: Optional[Path], random_seed: int = 42) -> Tuple[pd.DataFrame, SourceType]:
    if data_path is not None:
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        df = pd.read_csv(data_path)
        if df.empty:
            raise ValueError(f"Dataset is empty: {data_path}")
        return df, "csv"

    rng = np.random.default_rng(random_seed)
    rows = 750

    market_regime = rng.integers(0, 3, size=rows)
    momentum_5 = rng.normal(loc=0.0, scale=0.8, size=rows)
    momentum_20 = rng.normal(loc=0.0, scale=1.2, size=rows)
    volatility_10 = np.abs(rng.normal(loc=0.02, scale=0.01, size=rows))
    volume_zscore = rng.normal(loc=0.0, scale=1.0, size=rows)
    valuation_gap = rng.normal(loc=0.0, scale=1.4, size=rows)

    next_return = (
        0.025 * momentum_5
        + 0.032 * momentum_20
        - 0.9 * volatility_10
        + 0.012 * volume_zscore
        + 0.008 * valuation_gap
        + 0.02 * (market_regime == 2)
        + rng.normal(loc=0.0, scale=0.03, size=rows)
    )

    synthetic_df = pd.DataFrame(
        {
            "market_regime": market_regime,
            "momentum_5": momentum_5,
            "momentum_20": momentum_20,
            "volatility_10": volatility_10,
            "volume_zscore": volume_zscore,
            "valuation_gap": valuation_gap,
            "next_return": next_return,
        }
    )

    return synthetic_df, "synthetic"
