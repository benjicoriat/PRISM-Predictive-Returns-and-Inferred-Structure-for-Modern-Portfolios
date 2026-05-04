"""Orchestrator and Agent classes for the PRISM review team.

The orchestrator drives a panel of specialist :class:`Agent` instances,
each backed by an Anthropic Claude model. Specialists run in parallel
threads; their findings are reduced into a single consolidated report
by an orchestrator pass.

The implementation depends on the ``anthropic`` Python SDK and uses
prompt caching on the per-role artefact bundle so that repeated runs
(e.g. iterating on a single role) stay cheap.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .context import PROJECT_ROOT, load_for_role, render_artefacts

ROLES = (
    "code_reviewer",
    "math_reviewer",
    "paper_reviewer",
    "reproducibility_engineer",
    "quant_validator",
    "consistency_auditor",
    "documentation_reviewer",
)

DEFAULT_SPECIALIST_MODEL = "claude-sonnet-4-6"
DEFAULT_ORCHESTRATOR_MODEL = "claude-opus-4-7"


@dataclass
class Finding:
    """Structured issue produced by a specialist agent."""

    role: str
    severity: str  # critical | high | medium | low | nit
    title: str
    location: str
    body: str

    def render(self) -> str:
        return (
            f"[{self.severity.upper()}] {self.location} — {self.title}\n"
            f"  ({self.role})\n"
            f"  {self.body.strip()}"
        )


@dataclass
class Agent:
    """Single specialist reviewer.

    Each agent is parameterised by a role; the role determines (a) the
    system prompt loaded from ``agents/roles/<role>.md`` and (b) the
    artefact bundle loaded from :func:`context.load_for_role`.
    """

    role: str
    model: str = DEFAULT_SPECIALIST_MODEL
    max_tokens: int = 4096

    @property
    def system_prompt(self) -> str:
        path = Path(__file__).parent / "roles" / f"{self.role}.md"
        return path.read_text(encoding="utf-8")

    def review(self, client) -> str:
        """Run the review and return the raw response text."""
        artefacts = load_for_role(self.role)
        artefact_blob = render_artefacts(artefacts)
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": self.system_prompt,
                },
                {
                    "type": "text",
                    "text": (
                        "Project artefacts follow. Treat them as the only "
                        "source of truth.\n\n" + artefact_blob
                    ),
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Produce your review now. Follow the output "
                        "format from your role brief exactly."
                    ),
                }
            ],
        )
        return "".join(block.text for block in message.content if block.type == "text")


@dataclass
class Orchestrator:
    """Coordinates the panel and reduces findings into a single report."""

    specialist_model: str = DEFAULT_SPECIALIST_MODEL
    orchestrator_model: str = DEFAULT_ORCHESTRATOR_MODEL
    parallel: bool = True
    max_workers: int = 4
    output_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "agents" / "reports" / "review.md"
    )

    def _client(self):
        try:
            import anthropic  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "The 'anthropic' package is required. Install with: "
                "pip install anthropic"
            ) from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is not set."
            )
        return anthropic.Anthropic()

    def _build_agents(self, only: Optional[list[str]] = None) -> list[Agent]:
        roles = only if only else list(ROLES)
        return [Agent(role=r, model=self.specialist_model) for r in roles]

    def run(self, only: Optional[list[str]] = None) -> Path:
        """Run the full review pipeline and write the consolidated report.

        Returns the path to the written report.
        """
        client = self._client()
        agents = self._build_agents(only)
        reviews: dict[str, str] = {}

        def _run(agent: Agent) -> tuple[str, str]:
            return agent.role, agent.review(client)

        if self.parallel:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futs = [pool.submit(_run, a) for a in agents]
                for f in as_completed(futs):
                    role, text = f.result()
                    reviews[role] = text
        else:
            for a in agents:
                role, text = _run(a)
                reviews[role] = text

        consolidated = self._reduce(client, reviews)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(consolidated, encoding="utf-8")
        return self.output_path

    def _reduce(self, client, reviews: dict[str, str]) -> str:
        orchestrator_prompt = (
            Path(__file__).parent / "roles" / "orchestrator.md"
        ).read_text(encoding="utf-8")
        joined = "\n\n".join(
            f"### Specialist: {role}\n\n{text}" for role, text in reviews.items()
        )
        message = client.messages.create(
            model=self.orchestrator_model,
            max_tokens=8192,
            system=orchestrator_prompt,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Below are the raw reviews from each specialist. "
                        "Reconcile, deduplicate, and produce the final "
                        "Markdown report described in your brief.\n\n"
                        + joined
                    ),
                }
            ],
        )
        return "".join(b.text for b in message.content if b.type == "text")
