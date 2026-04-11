import shutil
from pathlib import Path


def clear_directory_contents(directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)

    removed_items = 0
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        removed_items += 1

    return removed_items
