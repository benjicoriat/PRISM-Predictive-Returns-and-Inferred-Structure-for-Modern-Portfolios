from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from prism.config import PipelineConfig
from prism.data import load_dataset
from prism.evaluation import RegressionMetrics, evaluate_predictions
from prism.features import build_feature_matrix, infer_target_column
from prism.model import train_regressor
from prism.visualization import (
    save_feature_importance,
    save_prediction_scatter,
    save_target_distribution,
)


@dataclass(frozen=True)
class PipelineResult:
    source_type: str
    target_column: str
    train_rows: int
    test_rows: int
    metrics: RegressionMetrics
    artifacts: List[Path]


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    raw_df, source_type = load_dataset(config.data_path, config.random_seed)
    target_column = infer_target_column(raw_df, config.target_column)
    x_df, y = build_feature_matrix(raw_df, target_column)

    if config.mini_mode and x_df.shape[0] > 2500:
        sampled_index = x_df.sample(n=2500, random_state=config.random_seed).index
        x_df = x_df.loc[sampled_index]
        y = y.loc[sampled_index]

    x_train, x_test, y_train, y_test = train_test_split(
        x_df,
        y,
        test_size=config.test_size,
        random_state=config.random_seed,
    )

    model = train_regressor(
        x_train,
        y_train,
        config.random_seed,
        mini_mode=config.mini_mode,
    )
    y_pred = model.predict(x_test)

    metrics = evaluate_predictions(y_test.to_numpy(), np.asarray(y_pred))

    image_dir = Path(config.image_dir)
    other_output_dir = Path(config.other_output_dir)
    other_output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: List[Path] = []
    artifacts.append(save_target_distribution(y, image_dir))
    artifacts.append(save_prediction_scatter(y_test.to_numpy(), np.asarray(y_pred), image_dir))

    feature_plot_path = save_feature_importance(model, x_df.columns.tolist(), image_dir)
    if feature_plot_path is not None:
        artifacts.append(feature_plot_path)

    metrics_csv = other_output_dir / "metrics_summary.csv"
    pd.DataFrame(
        [
            {
                "source_type": source_type,
                "target_column": target_column,
                "mini_mode": config.mini_mode,
                "train_rows": x_train.shape[0],
                "test_rows": x_test.shape[0],
                "r2": metrics.r2,
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "directional_accuracy": metrics.directional_accuracy,
            }
        ]
    ).to_csv(metrics_csv, index=False)
    artifacts.append(metrics_csv)

    predictions_csv = other_output_dir / "predictions_preview.csv"
    pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted": np.asarray(y_pred),
        }
    ).head(500).to_csv(predictions_csv, index=False)
    artifacts.append(predictions_csv)

    return PipelineResult(
        source_type=source_type,
        target_column=target_column,
        train_rows=int(x_train.shape[0]),
        test_rows=int(x_test.shape[0]),
        metrics=metrics,
        artifacts=artifacts,
    )
