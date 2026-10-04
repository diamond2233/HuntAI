"""Benchmarks the full HuntAI pipeline against real sources and real OpenAI calls.

Run manually with:
    python scripts/benchmark.py                     -> saves results/baseline.json
    python scripts/benchmark.py after_concurrency.json -> saves results/after_concurrency.json

This script does not change any pipeline code. It calls the same collectors
and pipeline functions main.py uses, and only wraps (not modifies) the
OpenAI client so it can read each real response's token usage.

Apify is skipped even if enabled in config/profile.yaml -- LinkedIn is
currently blocking that collector entirely, and this benchmark only
measures the sources that actually return jobs right now.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import yaml
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from collectors.greenhouse import GreenhouseCollector  # noqa: E402
from collectors.lever import LeverCollector  # noqa: E402
from models.job import Job  # noqa: E402
from pipeline.dedup import deduplicate  # noqa: E402
from pipeline.filters import apply_filters, is_excluded_company, matches_location, matches_role  # noqa: E402
from pipeline.matcher import match_jobs  # noqa: E402
from pipeline.ranker import rank_jobs  # noqa: E402

load_dotenv()

# Verified at platform.openai.com/docs/pricing on 2026-10-04, standard
# (non-batch) rate. Only accurate for this exact model -- if OPENAI_MODEL
# changes, update these before trusting the cost estimate.
PRICED_MODEL = "gpt-4.1-mini"
PRICE_PER_1M_INPUT_TOKENS = 0.40
PRICE_PER_1M_OUTPUT_TOKENS = 1.60


def _load_profile() -> dict:
    with open(PROJECT_ROOT / "config" / "profile.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _collect(profile: dict) -> tuple[list[Job], dict[str, int]]:
    """Run the configured Greenhouse and Lever collectors.

    Mirrors collect_node's logic, but only for Greenhouse/Lever -- see the
    module docstring for why Apify is left out.
    """
    sources = profile.get("sources") or {}
    jobs: list[Job] = []
    jobs_per_source = {"greenhouse": 0, "lever": 0}

    greenhouse_config = sources.get("greenhouse") or {}
    if greenhouse_config.get("enabled"):
        for board in greenhouse_config.get("boards") or []:
            token = board.get("token")
            if not token:
                continue
            collector = GreenhouseCollector(board_token=token, company=board.get("company"))
            board_jobs = collector.collect()
            jobs.extend(board_jobs)
            jobs_per_source["greenhouse"] += len(board_jobs)

    lever_config = sources.get("lever") or {}
    if lever_config.get("enabled"):
        for company in lever_config.get("companies") or []:
            slug = company.get("slug")
            if not slug:
                continue
            collector = LeverCollector(company_slug=slug, company=company.get("company"))
            company_jobs = collector.collect()
            jobs.extend(company_jobs)
            jobs_per_source["lever"] += len(company_jobs)

    return jobs, jobs_per_source


def _match_with_usage_tracking(jobs: list[Job], profile: dict) -> tuple[list, list[dict]]:
    """Call the real match_jobs() unmodified, recording each call's token usage.

    The OpenAI client is wrapped (not changed) so usage can be read off each
    real response -- pipeline/matcher.py itself is untouched by this script.
    """
    real_client = OpenAI()
    usage_log: list[dict] = []
    original_parse = real_client.chat.completions.parse

    def instrumented_parse(*args, **kwargs):
        completion = original_parse(*args, **kwargs)
        if completion.usage:
            usage_log.append(
                {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens,
                }
            )
        return completion

    real_client.chat.completions.parse = instrumented_parse

    with patch("pipeline.matcher.OpenAI", return_value=real_client):
        matched_jobs = match_jobs(jobs, profile)

    return matched_jobs, usage_log


# Substrings used to flag a rejected title as "plausibly a real tech role"
# -- purely for spotting false negatives in the role filter. Checked as a
# plain lowercase substring (not whole-word), since this is just a human
# sanity check, not a filtering rule.
TECH_KEYWORDS_IN_REJECTED_TITLES = [
    "engineer", "engineering", "developer", "development", "scientist",
    "analyst", "sde", "technical", "software", "data", "ml", "ai",
    "backend", "frontend", "platform", "devops", "security", "qa",
]


def _looks_like_a_tech_title(title: str) -> bool:
    title_lower = title.lower()
    return any(keyword in title_lower for keyword in TECH_KEYWORDS_IN_REJECTED_TITLES)


def _diagnose_filter_rejections(jobs: list[Job], profile: dict) -> tuple[dict[str, int], dict]:
    """Classify why each job was rejected, using filters.py's own functions.

    Re-applies is_excluded_company/matches_role/matches_location in the same
    order apply_filters() does, so a job ends up counted under whichever
    check it actually failed first. This is read-only diagnostics -- it
    doesn't change what apply_filters() itself does.

    Returns (rejections, extra) where rejections is {reason: count} (saved
    to the benchmark JSON) and extra holds the detailed, print-only
    breakdowns used to investigate the role filter specifically.
    """
    excluded_companies = profile.get("excluded_companies") or []
    roles = profile.get("roles") or []
    locations = profile.get("locations") or []

    rejections = Counter()
    rejected_location_strings = Counter()
    rejected_role_titles = Counter()
    tech_looking_titles_rejected_for_role = Counter()
    per_company = {}

    for job in jobs:
        company_stats = per_company.setdefault(
            job.company, {"collected": 0, "rejected_for_role": 0, "kept": 0}
        )
        company_stats["collected"] += 1

        if is_excluded_company(job.company, excluded_companies):
            rejections["excluded_company"] += 1
            continue
        if not matches_role(job.title, roles):
            rejections["role"] += 1
            rejected_role_titles[job.title] += 1
            company_stats["rejected_for_role"] += 1
            if _looks_like_a_tech_title(job.title):
                tech_looking_titles_rejected_for_role[job.title] += 1
            continue
        if not matches_location(job.location, locations):
            rejections["location"] += 1
            rejected_location_strings[job.location or "(none)"] += 1
            continue

        company_stats["kept"] += 1

    extra = {
        "rejected_location_strings": rejected_location_strings,
        "rejected_role_titles": rejected_role_titles,
        "tech_looking_titles_rejected_for_role": tech_looking_titles_rejected_for_role,
        "per_company": per_company,
    }
    return dict(rejections), extra


def _estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    if model != PRICED_MODEL:
        return None
    return round(
        (prompt_tokens / 1_000_000) * PRICE_PER_1M_INPUT_TOKENS
        + (completion_tokens / 1_000_000) * PRICE_PER_1M_OUTPUT_TOKENS,
        6,
    )


def run_benchmark() -> tuple[dict, dict]:
    """Runs the pipeline once and returns (results, rejection_diagnostics).

    rejection_diagnostics is print-only (see _diagnose_filter_rejections) and
    isn't part of the saved JSON -- it's returned so main() can print it.
    """
    profile = _load_profile()
    stage_seconds = {}
    total_start = time.perf_counter()

    start = time.perf_counter()
    jobs, jobs_per_source = _collect(profile)
    stage_seconds["collect"] = time.perf_counter() - start

    start = time.perf_counter()
    filtered_jobs = apply_filters(jobs, profile)
    stage_seconds["filter"] = time.perf_counter() - start

    filter_rejections, rejection_diagnostics = _diagnose_filter_rejections(jobs, profile)

    start = time.perf_counter()
    deduplicated_jobs = deduplicate(filtered_jobs)
    stage_seconds["dedup"] = time.perf_counter() - start

    start = time.perf_counter()
    matched_jobs, usage_log = _match_with_usage_tracking(deduplicated_jobs, profile)
    stage_seconds["match"] = time.perf_counter() - start

    start = time.perf_counter()
    ranked_jobs = rank_jobs(matched_jobs)
    stage_seconds["rank"] = time.perf_counter() - start

    stage_seconds["total"] = time.perf_counter() - total_start

    prompt_tokens = sum(u["prompt_tokens"] for u in usage_log)
    completion_tokens = sum(u["completion_tokens"] for u in usage_log)
    total_tokens = sum(u["total_tokens"] for u in usage_log)
    model = os.getenv("OPENAI_MODEL")
    estimated_cost_usd = _estimate_cost_usd(model, prompt_tokens, completion_tokens)

    return {
        "jobs_collected_per_source": jobs_per_source,
        "jobs_collected_total": len(jobs),
        "jobs_after_filter": len(filtered_jobs),
        "filter_rejections": filter_rejections,
        "jobs_after_dedup": len(deduplicated_jobs),
        "openai_calls": len(usage_log),
        "openai_model": model,
        "openai_tokens": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "estimated_cost_usd": estimated_cost_usd,
        "cost_pricing_note": (
            f"Priced using {PRICED_MODEL} rates (${PRICE_PER_1M_INPUT_TOKENS}/1M input, "
            f"${PRICE_PER_1M_OUTPUT_TOKENS}/1M output tokens), verified at "
            "platform.openai.com/docs/pricing on 2026-10-04."
            if estimated_cost_usd is not None
            else f"No cost estimate: configured model '{model}' is not {PRICED_MODEL}, "
            "the only model this script has verified pricing for."
        ),
        "final_ranked_jobs": len(ranked_jobs),
        "stage_seconds": {stage: round(seconds, 3) for stage, seconds in stage_seconds.items()},
    }, rejection_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the HuntAI pipeline end-to-end.")
    parser.add_argument(
        "output_name",
        nargs="?",
        default="baseline.json",
        help="Filename to save results under in results/ (default: baseline.json)",
    )
    args = parser.parse_args()

    results, diagnostics = run_benchmark()

    output_path = PROJECT_ROOT / "results" / args.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nSaved to {output_path}")

    print("\nTop 25 location strings among jobs rejected for location:")
    for location, count in diagnostics["rejected_location_strings"].most_common(25):
        print(f"  {count:>4}  {location}")

    print("\nTop 40 most common rejected-for-role titles:")
    for title, count in diagnostics["rejected_role_titles"].most_common(40):
        print(f"  {count:>4}  {title}")

    print("\nJobs per company (collected / rejected for role / kept):")
    for company, stats in sorted(diagnostics["per_company"].items()):
        print(f"  {company:<30} collected={stats['collected']:>4}  rejected_for_role={stats['rejected_for_role']:>4}  kept={stats['kept']:>4}")

    tech_looking = diagnostics["tech_looking_titles_rejected_for_role"]
    print(
        f"\nRejected-for-role titles that look like tech roles "
        f"(contain engineer/developer/scientist/analyst/sde/technical/software/"
        f"data/ml/ai/backend/frontend/platform/devops/security/qa) -- {sum(tech_looking.values())} total:"
    )
    for title, count in tech_looking.most_common():
        print(f"  {count:>4}  {title}")


if __name__ == "__main__":
    main()
