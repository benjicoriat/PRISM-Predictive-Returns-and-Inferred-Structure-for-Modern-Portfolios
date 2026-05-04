"""PRISM agent team for project review.

The :class:`Orchestrator` dispatches a panel of specialist agents
(code, math, paper, reproducibility, quant, consistency, docs) over the
PRISM artefacts and produces a consolidated review report.
"""

from .orchestrator import Agent, Orchestrator, Finding

__all__ = ["Agent", "Orchestrator", "Finding"]
