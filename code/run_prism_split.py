#!/usr/bin/env python3
"""Run the split PRISM v3 pipeline cells in order within one shared namespace."""

from pathlib import Path


def load_cell_order(order_file: Path) -> list[str]:
    lines = [line.strip() for line in order_file.read_text(encoding="utf-8-sig").splitlines()]
    return [line for line in lines if line.endswith(".py")]


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    order_file = base_dir / "cell_order.txt"
    cell_files = load_cell_order(order_file)

    namespace: dict[str, object] = {"__name__": "__main__"}
    for cell_name in cell_files:
        cell_path = base_dir / cell_name
        source = cell_path.read_text(encoding="utf-8-sig")
        namespace["__file__"] = str(cell_path)
        exec(compile(source, str(cell_path), "exec"), namespace)


if __name__ == "__main__":
    main()
