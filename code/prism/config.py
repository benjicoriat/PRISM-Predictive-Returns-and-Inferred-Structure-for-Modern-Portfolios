from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PipelineConfig:
    data_path: Optional[Path] = None
    target_column: Optional[str] = None
    test_size: float = 0.2
    random_seed: int = 42
    image_dir: Path = Path("images")
    other_output_dir: Path = Path("other_outputs")
    mini_mode: bool = False
