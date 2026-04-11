import argparse
from pathlib import Path
from typing import Optional

from prism.config import PipelineConfig
from prism.io_utils import clear_directory_contents
from prism.pipeline import run_pipeline


def _resolve_data_path(raw_data_path: Optional[str], repo_root: Path) -> Optional[Path]:
    if raw_data_path is None:
        return None

    candidate = Path(raw_data_path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate

    return candidate.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PRISM predictive return pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional CSV dataset path. If omitted, synthetic data is generated.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Optional target column name to predict.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction in [0, 1].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for synthetic data and model training.",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Run a very fast execution using lightweight model settings.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete all contents in images and other_outputs, then exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    image_dir = repo_root / "images"
    other_output_dir = repo_root / "other_outputs"

    if args.cleanup:
        removed_images = clear_directory_contents(image_dir)
        removed_other = clear_directory_contents(other_output_dir)
        print(
            "Cleanup complete. "
            f"Removed {removed_images} item(s) from {image_dir} and "
            f"{removed_other} item(s) from {other_output_dir}."
        )
        return 0

    config = PipelineConfig(
        data_path=_resolve_data_path(args.data, repo_root),
        target_column=args.target,
        test_size=args.test_size,
        random_seed=args.seed,
        image_dir=image_dir,
        other_output_dir=other_output_dir,
        mini_mode=args.mini,
    )

    result = run_pipeline(config)

    print("PRISM pipeline completed")
    print(f"Mode: {'mini' if args.mini else 'standard'}")
    print(f"Source: {result.source_type}")
    print(f"Target: {result.target_column}")
    print(f"Train rows: {result.train_rows} | Test rows: {result.test_rows}")
    print(
        "Metrics: "
        f"R2={result.metrics.r2:.4f}, "
        f"MAE={result.metrics.mae:.6f}, "
        f"RMSE={result.metrics.rmse:.6f}, "
        f"DirectionalAcc={result.metrics.directional_accuracy:.4f}"
    )
    print("Artifacts:")
    for artifact in result.artifacts:
        print(f" - {artifact}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
