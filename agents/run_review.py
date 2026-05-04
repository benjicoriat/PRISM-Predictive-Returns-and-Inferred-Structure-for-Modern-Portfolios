"""CLI entry point for the PRISM agent review.

Usage::

    # Run the full panel
    python -m agents.run_review

    # Run a single role
    python -m agents.run_review --only code_reviewer

    # Force sequential execution (useful for debugging API errors)
    python -m agents.run_review --serial

The script writes a consolidated Markdown report to
``agents/reports/review.md`` and prints the path on stdout.

Set ``ANTHROPIC_API_KEY`` in the environment first.
"""

from __future__ import annotations

import argparse
import sys

from .orchestrator import (
    DEFAULT_ORCHESTRATOR_MODEL,
    DEFAULT_SPECIALIST_MODEL,
    ROLES,
    Orchestrator,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Run the PRISM agent review panel.")
    p.add_argument(
        "--only",
        action="append",
        choices=ROLES,
        help="Restrict the panel to one or more roles (repeatable).",
    )
    p.add_argument(
        "--specialist-model",
        default=DEFAULT_SPECIALIST_MODEL,
        help=f"Model for specialist agents (default: {DEFAULT_SPECIALIST_MODEL}).",
    )
    p.add_argument(
        "--orchestrator-model",
        default=DEFAULT_ORCHESTRATOR_MODEL,
        help=f"Model for the orchestrator (default: {DEFAULT_ORCHESTRATOR_MODEL}).",
    )
    p.add_argument(
        "--serial",
        action="store_true",
        help="Disable parallel specialist execution.",
    )
    args = p.parse_args()

    orch = Orchestrator(
        specialist_model=args.specialist_model,
        orchestrator_model=args.orchestrator_model,
        parallel=not args.serial,
    )
    out = orch.run(only=args.only)
    print(f"Report written to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
