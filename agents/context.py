"""Project-context loader for PRISM agents.

Each role only sees the artefacts it actually needs. Files are read once
and packaged into a list of ``(label, content)`` tuples that the
orchestrator can drop into a prompt with ``cache_control`` for
prompt-caching.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Artefact:
    label: str
    path: Path

    def read(self) -> str:
        return self.path.read_text(encoding="utf-8-sig", errors="replace")


def _glob(*patterns: str) -> list[Artefact]:
    seen: dict[Path, Artefact] = {}
    for pattern in patterns:
        for p in sorted(PROJECT_ROOT.glob(pattern)):
            if p.is_file() and p not in seen:
                rel = p.relative_to(PROJECT_ROOT).as_posix()
                seen[p] = Artefact(label=rel, path=p)
    return list(seen.values())


# ── Per-role artefact bundles ────────────────────────────────────────────
ROLE_ARTEFACTS: dict[str, tuple[str, ...]] = {
    "code_reviewer": (
        "code/cell_*.py",
        "code/run_prism_split.py",
        "code/generate_missing_images.py",
        "code/00_header.txt",
        "code/cell_order.txt",
    ),
    "math_reviewer": (
        "Latex/maths.tex",
        "Latex/PRISM.tex",
    ),
    "paper_reviewer": (
        "Latex/PRISM.tex",
        "outputs/performance_table.txt",
        "outputs/performance_table.csv",
    ),
    "reproducibility_engineer": (
        "README.md",
        "requirements.txt",
        "code/cell_*.py",
        "code/run_prism_split.py",
        "code/generate_missing_images.py",
    ),
    "quant_validator": (
        "code/cell_*.py",
        "Latex/PRISM.tex",
        "Latex/maths.tex",
        "outputs/performance_table.txt",
    ),
    "consistency_auditor": (
        "Latex/PRISM.tex",
        "Latex/maths.tex",
        "Latex/code.tex",
        "code/cell_*.py",
        "code/cell_order.txt",
        "outputs/performance_table.csv",
    ),
    "documentation_reviewer": (
        "README.md",
        "code/README.txt",
        "code/run_prism_split.py",
        "code/cell_01.py",
        "code/cell_02.py",
        "code/cell_03.py",
        "Latex/PRISM.tex",
    ),
}


def load_for_role(role: str) -> list[Artefact]:
    if role not in ROLE_ARTEFACTS:
        raise KeyError(f"Unknown role: {role}")
    return _glob(*ROLE_ARTEFACTS[role])


def render_artefacts(artefacts: Iterable[Artefact]) -> str:
    """Render artefacts into a single delimited blob suitable for a prompt."""
    parts = []
    for a in artefacts:
        parts.append(f"<<< FILE: {a.label} >>>")
        parts.append(a.read())
        parts.append(f"<<< END FILE: {a.label} >>>")
    return "\n".join(parts)
