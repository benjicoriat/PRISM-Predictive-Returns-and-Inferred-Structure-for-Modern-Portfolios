# PRISM Agent Team

A panel of specialist review agents that read the PRISM project end-to-end
(code, paper, math) and produce a single consolidated review report.

## The team

| Role | What it reads | What it checks |
|---|---|---|
| `code_reviewer` | `code/cell_*.py`, runners | Bugs, numerical hazards, silent failures, hardcoded magic, unsafe installs |
| `math_reviewer` | `Latex/maths.tex`, equations in `PRISM.tex` | Definitions, dimensions, NIW posterior, Sinkhorn, confidence score |
| `paper_reviewer` | `Latex/PRISM.tex`, results tables | Claims vs evidence, internal consistency, references, captions |
| `reproducibility_engineer` | `README.md`, `requirements.txt`, code | Dependency completeness, deprecated APIs, determinism, time bombs |
| `quant_validator` | code + paper + math | Look-ahead bias, cost realism, Sharpe accounting, vol-targeting |
| `consistency_auditor` | all three artefacts together | Drift between paper, math, and code |
| `documentation_reviewer` | `README.md`, `code/README.txt`, docstrings | Quick start, file map, promised outputs, deliverable alignment |
| **`orchestrator`** | the panel's findings | Dedupe, reconcile, prioritise, emit one report |

Each specialist has a system prompt under `agents/roles/<role>.md` and a
curated artefact bundle defined in `agents/context.py`.

## Running it

```bash
export ANTHROPIC_API_KEY=...
pip install anthropic        # the only extra dep
python -m agents.run_review  # full panel, parallel
```

Single role:

```bash
python -m agents.run_review --only code_reviewer
```

Output is written to `agents/reports/review.md`.

## Architecture

```
                    ┌──────────────────────┐
                    │    Orchestrator      │
                    │  (claude-opus-4-7)   │
                    └──────────┬───────────┘
                               │ reduce
        ┌──────────┬───────────┼───────────┬──────────┐
        ▼          ▼           ▼           ▼          ▼
    ┌────────┐┌────────┐ ┌────────┐  ┌────────┐ ┌────────┐
    │ code   ││ math   │ │ paper  │  │ repro  │ │  quant │  ...
    │reviewer││reviewer│ │reviewer│  │  eng   │ │validato│
    └────────┘└────────┘ └────────┘  └────────┘ └────────┘
        ▲          ▲           ▲           ▲          ▲
        └──────────┴───────────┴───────────┴──────────┘
                    cached per-role artefacts
```

Specialists run in parallel via a thread pool; their per-role artefact
blob is cached with `cache_control: ephemeral`, so re-running a single
role after a fix is cheap. The orchestrator reduces all specialist
outputs into one prioritised punch list.

## Use from Claude Code

The same roles are available as Claude Code subagents under
`.claude/agents/`. Invoke them with the `Agent` tool, e.g.
`subagent_type: prism-code-reviewer`.

## Extending

Add a new role:

1. Drop a new `agents/roles/<role>.md` system prompt.
2. Register the artefact bundle in `agents/context.py::ROLE_ARTEFACTS`.
3. Add the role name to `agents/orchestrator.py::ROLES`.
4. (Optional) Mirror it under `.claude/agents/prism-<role>.md`.
