# HuntAI

HuntAI is a terminal-based, configuration-driven job discovery system. It
discovers software engineering jobs from job boards and ranks them according
to a user's preferences, defined in a single YAML profile.

## Architecture

```
config/profile.yaml
        |
        v
LangGraph orchestrator
        |
        v
   job collection
        |
        v
   normalization
        |
        v
   hard filtering
        |
        v
  deduplication
        |
        v
LLM-based semantic matching
        |
        v
      ranking
        |
        v
Markdown / CSV / JSON export
```

Each stage reads and writes a shared pipeline state. Deterministic stages
(filtering, deduplication, ranking, export) never call the LLM. Only the
matching stage calls OpenAI, and it does so with a structured (Pydantic)
input and output, not free-form text.

## Tech stack

- Python 3.11+
- LangGraph — orchestrates the pipeline as a graph of nodes with shared state
- Pydantic — validates all structured data (jobs, matches) as it moves through
  the pipeline
- OpenAI API / OpenAI Python SDK — the runtime LLM, used only for semantic
  job-to-profile matching
- PyYAML — loads the user profile from `config/profile.yaml`
- python-dotenv — loads OpenAI credentials from `.env`
- pytest — tests

## Role of each component

- **LangGraph**: defines the pipeline as an explicit graph
  (`collect -> filter -> deduplicate -> match -> rank -> export`) so each
  stage is a separate, replaceable node operating on shared state.
- **Pydantic**: defines the `Job` model (a normalized job posting) and the
  `JobMatch` model (a structured LLM matching result), so no unstructured
  data flows through the pipeline.
- **OpenAI**: used exclusively by the matching stage to semantically compare
  a job description against the user's profile and produce a structured
  `JobMatch`.
- **Collectors**: `collectors/base.py` defines a common interface
  (`collect() -> list[Job]`). `GreenhouseCollector` and `LeverCollector` will
  each implement this interface and convert their source's raw API response
  into `Job` objects, so the rest of the pipeline never needs to know which
  source a job came from.

## Project structure

```
huntAI/
├── config/profile.yaml       # user preferences (roles, locations, skills, etc.)
├── models/job.py              # Job and JobMatch Pydantic models
├── collectors/                # base interface + Greenhouse/Lever placeholders
├── pipeline/                  # LangGraph state, graph, and stage placeholders
├── exporters/                 # Markdown/CSV/JSON export placeholders
├── tests/test_basic.py        # tests for the Pydantic models
├── output/                    # generated export files (gitignored)
├── main.py                    # minimal entry point
├── requirements.txt
├── .env                       # OpenAI credentials (not committed)
└── .gitignore
```

## Current implementation status

**Implemented:**
- `Job` and `JobMatch` Pydantic models
- `BaseCollector` interface
- `GreenhouseCollector` / `LeverCollector` placeholder classes (raise
  `NotImplementedError`)
- Example `config/profile.yaml`
- `PipelineState` (TypedDict) for LangGraph
- LangGraph skeleton with all six nodes wired in sequence, each currently a
  pass-through
- Placeholder functions in `pipeline/filters.py`, `pipeline/dedup.py`,
  `pipeline/matcher.py`, `pipeline/ranker.py`
- Placeholder functions in `exporters/markdown.py`, `exporters/csv.py`,
  `exporters/json.py`
- `main.py` that loads the profile and runs the empty graph end-to-end
- Basic tests for the Pydantic models

**NOT implemented (future stages):**
- Real Greenhouse/Lever API requests
- Hard filtering logic
- Deduplication logic
- OpenAI-based semantic matching
- Ranking logic
- Markdown/CSV/JSON export logic
- Any network calls or LLM calls

## Setup

```
pip install -r requirements.txt
```

Fill in `OPENAI_API_KEY` and `OPENAI_MODEL` in `.env` before later stages
that require OpenAI access. No key is needed for the current foundation.

## Running tests

```
pytest
```
