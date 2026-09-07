# HuntAI

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-126%20passing-brightgreen)
![Pipeline](https://img.shields.io/badge/pipeline-collect%20%E2%86%92%20filter%20%E2%86%92%20dedup%20%E2%86%92%20match%20%E2%86%92%20rank%20%E2%86%92%20export-informational)

HuntAI is a terminal-based, configuration-driven job discovery system. It
collects software engineering jobs from job boards, filters and deduplicates
them, uses an LLM to semantically match them against a user's profile, ranks
the results, and exports them to Markdown/CSV/JSON — all driven by a single
YAML profile, no code changes required to reconfigure.

## Architecture

```
config/profile.yaml
        |
        v
LangGraph orchestrator
        |
        v
   job collection        <- Greenhouse, Lever, Apify (LinkedIn)
        |
        v
   hard filtering         <- excluded companies, role/title, location
        |
        v
  deduplication           <- by normalized URL
        |
        v
LLM-based semantic matching  <- OpenAI, one call per job, structured output
        |
        v
      ranking             <- sorted by match score
        |
        v
Markdown / CSV / JSON export
```

Each stage reads and writes a shared pipeline state. Deterministic stages
(filtering, deduplication, ranking, export) never call an LLM. Only the
matching stage calls OpenAI, and it does so with a structured (Pydantic)
input and output, not free-form text.

## Tech stack

- Python 3.11+
- LangGraph — orchestrates the pipeline as a graph of nodes with shared state
- Pydantic — validates all structured data (jobs, matches) as it moves through
  the pipeline
- OpenAI API / OpenAI Python SDK — the runtime LLM, used only for semantic
  job-to-profile matching
- apify-client — runs the Apify LinkedIn job-search actor
- PyYAML — loads the user profile from `config/profile.yaml`
- python-dotenv — loads API credentials from `.env`
- pytest — tests

## Job sources

| Source     | How it works                                                        |
|------------|----------------------------------------------------------------------|
| Greenhouse | Public job-board API, one or more company board tokens               |
| Lever      | Public job-board API, one or more company slugs                      |
| Apify      | Searches LinkedIn via the `jobsapi/linkedin-jobs-search-scraper` actor, using the roles/locations from your profile |

Each source is independently enabled/disabled in `config/profile.yaml` under
`sources:`.

> **Note:** Apify is currently disabled by default in `config/profile.yaml`
> because LinkedIn is blocking the actor's guest search entirely
> (`NO_SEARCH_RESULTS` on every location) — an external service issue, not a
> bug in this repo. Set `sources.apify.enabled: true` once that clears up, or
> if you're using a different Apify actor/proxy configuration.

## Project structure

```
HuntAI/
├── config/profile.yaml    # user preferences: roles, locations, skills, excluded companies, sources
├── models/job.py          # Job and JobMatch Pydantic models
├── collectors/            # Greenhouse, Lever, and Apify collectors (BaseCollector interface)
├── pipeline/               # LangGraph state, graph, filters, dedup, matcher, ranker
├── exporters/              # Markdown/CSV/JSON export
├── tests/                  # pytest suite (126 tests, all mocked — no real network/LLM calls)
├── output/                 # generated export files (gitignored)
├── main.py                 # entry point: loads profile, runs the graph
├── requirements.txt
├── .env.example             # template for required environment variables
└── .gitignore
```

## Setup

1. **Clone and install dependencies**

   ```bash
   git clone https://github.com/diamond2233/HuntAI.git
   cd HuntAI
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure credentials**

   Copy `.env.example` to `.env` and fill in your keys:

   ```bash
   cp .env.example .env
   ```

   ```
   OPENAI_API_KEY=your-key-here
   OPENAI_MODEL=gpt-4o-mini        # or whichever model you have access to
   APIFY_API_TOKEN=your-token-here # only needed if sources.apify.enabled is true
   ```

   `.env` is gitignored — never commit real keys.

3. **Configure your job search**

   Edit `config/profile.yaml`: your target roles, locations, experience
   range, skills, companies to exclude, and which sources
   (`greenhouse` / `lever` / `apify`) to enable and how to configure them
   (board tokens, company slugs).

4. **Run it**

   ```bash
   python main.py
   ```

   This runs the full pipeline and writes `output/jobs.md`, `output/jobs.csv`,
   and `output/jobs.json` — your ranked job matches.

## Running tests

The entire test suite runs offline against mocks — no real HTTP, Apify, or
OpenAI calls are made.

```bash
pytest
# or, if `pytest` isn't on PATH:
python -m pytest -v
```
