from pathlib import Path
from typing import Optional, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_output_dir(image_dir: Path) -> Path:
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir


def save_target_distribution(y: pd.Series, image_dir: Path) -> Path:
    output_dir = _ensure_output_dir(image_dir)
    output_path = output_dir / "target_distribution.png"

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(y, bins=40, color="#2E86AB", edgecolor="white", alpha=0.9)
    ax.set_title("Target Distribution")
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_prediction_scatter(y_true: np.ndarray, y_pred: np.ndarray, image_dir: Path) -> Path:
    output_dir = _ensure_output_dir(image_dir)
    output_path = output_dir / "actual_vs_predicted.png"

    min_v = float(min(np.min(y_true), np.min(y_pred)))
    max_v = float(max(np.max(y_true), np.max(y_pred)))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(y_true, y_pred, alpha=0.55, color="#F18F01", edgecolors="none")
    ax.plot([min_v, max_v], [min_v, max_v], color="#C73E1D", linewidth=2)
    ax.set_title("Actual vs Predicted Returns")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_feature_importance(
    model: object,
    feature_names: Sequence[str],
    image_dir: Path,
    top_n: int = 15,
) -> Optional[Path]:
    if not hasattr(model, "feature_importances_"):
        return None

    importances = np.asarray(getattr(model, "feature_importances_"))
    if importances.size == 0:
        return None

    output_dir = _ensure_output_dir(image_dir)
    output_path = output_dir / "feature_importance.png"

    ranked_indices = np.argsort(importances)[-top_n:]
    ranked_values = importances[ranked_indices]
    ranked_features = np.asarray(feature_names)[ranked_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(np.arange(len(ranked_values)), ranked_values, color="#2A9D8F")
    ax.set_yticks(np.arange(len(ranked_values)))
    ax.set_yticklabels(ranked_features)
    ax.set_title(f"Top {len(ranked_values)} Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return output_path
