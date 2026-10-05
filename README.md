# HuntAI

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-185%20passing-brightgreen)
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

## Limitations

- Sources: public Greenhouse and Lever boards only.
- Match scores are LLM judgments and have not been validated against
  human labels.
- LLM output is not perfectly deterministic even at temperature 0 (see
  `docs/H1_EXPLAINED.md`'s H1i section).
- The seniority filter is title-based, so it can miss senior roles with
  plain titles and can wrongly drop some good ones.

## Results

Real numbers from actual runs, saved under `results/`.

**Filter funnel** (`results/final_run.json`, full uncapped run) — of all
jobs collected from the configured Greenhouse/Lever boards, how many
survive each deterministic filter, in the order they're applied:

| Jobs collected | Rejected: company | Rejected: role | Rejected: seniority | Rejected: location | Jobs kept |
|---|---|---|---|---|---|
| 2234 | 0 | 1484 | 644 | 93 | 13 |

**Same full run — matching stage** (`results/final_run.json`):

| OpenAI calls | Skipped | Dropped matched skills | Dropped missing skills | Tokens | Cost | Total seconds |
|---|---|---|---|---|---|---|
| 13 | 0 | 1 | 0 | 35406 | $0.017156 | 39.355 |

**Matching all 165 filtered jobs at 1 / 4 / 8 workers** (`results/h1e_ratelimit.json`,
one run each, after the rate-limit fix):

| Workers | Seconds | Matched | Skipped | 429s | Tokens | Cost |
|---|---|---|---|---|---|---|
| 1 | 298.053 | 165 | 0 | 0 | 400613 | $0.195253 |
| 4 | 78.076 | 165 | 0 | 0 | 400667 | $0.195339 |
| 8 | 66.73 | 164 | 1 | 51 | 397307 | $0.193638 |

**60-job benchmark, 1 vs 8 workers, median of 3 runs** (`results/h1b_benchmark.json`):

| Workers | Seconds | Matched | Skipped | Tokens | Cost |
|---|---|---|---|---|---|
| 1 | 107.316 | 60 | 0 | 130570 | $0.064204 |
| 8 | 15.904 | 60 | 0 | 130691 | $0.064398 |

Beyond about 4 workers, OpenAI rate limits, not thread count, limit the speed.

## Project structure

```
HuntAI/
├── config/profile.yaml    # user preferences: roles, locations, skills, excluded companies, sources
├── models/job.py          # Job and JobMatch Pydantic models
├── collectors/            # Greenhouse, Lever, and Apify collectors (BaseCollector interface)
├── pipeline/               # LangGraph state, graph, filters, dedup, matcher, ranker
├── exporters/              # Markdown/CSV/JSON/skipped-jobs-report export
├── scripts/                # benchmark.py (real-run benchmarking), check_boards.py (find new boards)
├── docs/H1_EXPLAINED.md    # plain-language changelog of the H1 reliability/quality work, with a TOC
├── results/                # saved output of real benchmark runs (jobs counts, tokens, cost, timings)
├── tests/                  # pytest suite (185 tests, all mocked — no real network/LLM calls)
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
