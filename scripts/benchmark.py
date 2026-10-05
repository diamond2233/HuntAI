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
from pipeline.filters import (  # noqa: E402
    DEFAULT_EXCLUDE_TITLE_WORDS,
    apply_filters,
    has_excluded_title_word,
    is_excluded_company,
    matches_location,
    matches_role,
)
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


def _match_with_usage_tracking(
    jobs: list[Job], profile: dict, max_workers: int = 4
) -> tuple[list, list[dict], dict]:
    """Call the real match_jobs() unmodified, recording each call's token usage.

    The OpenAI client is wrapped (not changed) so usage can be read off each
    real response -- pipeline/matcher.py itself is untouched by this script.
    Returns (matched_jobs, usage_log, stats) where stats is match_jobs()'s
    own stats dict (currently just {"skipped_jobs": N}).
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

    stats: dict = {}
    with patch("pipeline.matcher.OpenAI", return_value=real_client):
        matched_jobs = match_jobs(jobs, profile, max_workers=max_workers, stats=stats)

    return matched_jobs, usage_log, stats


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

    Re-applies is_excluded_company/matches_role/has_excluded_title_word/
    matches_location in the same order apply_filters() does, so a job ends
    up counted under whichever check it actually failed first. This is
    read-only diagnostics -- it doesn't change what apply_filters() itself
    does.

    Returns (rejections, extra) where rejections is {reason: count} (saved
    to the benchmark JSON) and extra holds the detailed, print-only
    breakdowns used to investigate the role filter specifically.
    """
    excluded_companies = profile.get("excluded_companies") or []
    roles = profile.get("roles") or []
    locations = profile.get("locations") or []
    exclude_title_words = profile.get("exclude_title_words", DEFAULT_EXCLUDE_TITLE_WORDS)

    rejections = Counter()
    rejected_location_strings = Counter()
    rejected_role_titles = Counter()
    rejected_seniority_titles = Counter()
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
        if has_excluded_title_word(job.title, exclude_title_words):
            rejections["seniority"] += 1
            rejected_seniority_titles[job.title] += 1
            continue
        if not matches_location(job.location, locations):
            rejections["location"] += 1
            rejected_location_strings[job.location or "(none)"] += 1
            continue

        company_stats["kept"] += 1

    extra = {
        "rejected_location_strings": rejected_location_strings,
        "rejected_role_titles": rejected_role_titles,
        "rejected_seniority_titles": rejected_seniority_titles,
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


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def run_fair_comparison(workers: int, runs: int, max_jobs: int, output_name: str) -> None:
    """Collects and filters ONCE, then re-runs only the matching stage `runs`
    times at a fixed `workers` count, on the same (capped, stable-order) job
    list -- so repeated invocations with different `workers` values are
    comparing apples to apples, not different random jobs.

    Writes/merges into results/<output_name> under "by_workers"[str(workers)],
    so running this once with workers=1 and again with workers=8 (both with
    the same --runs/--max-jobs) builds up one file with both results side by
    side, and the second invocation prints the full comparison table.
    """
    profile = _load_profile()

    print("Collecting and filtering once (shared across all worker counts)...")
    jobs, jobs_per_source = _collect(profile)
    filtered_jobs = apply_filters(jobs, profile)
    filter_rejections, _diagnostics = _diagnose_filter_rejections(jobs, profile)
    deduplicated_jobs = deduplicate(filtered_jobs)

    jobs_after_filter = len(deduplicated_jobs)
    print(f"Jobs collected: {len(jobs)}  |  Jobs after filter+dedup: {jobs_after_filter}")

    output_path = PROJECT_ROOT / "results" / output_name
    if jobs_after_filter < 30:
        print(
            f"\nOnly {jobs_after_filter} jobs survived the filter (< 30) -- "
            "stopping before any OpenAI calls, as instructed."
        )
        results = {
            "jobs_collected_per_source": jobs_per_source,
            "jobs_collected_total": len(jobs),
            "jobs_after_filter": jobs_after_filter,
            "filter_rejections": filter_rejections,
            "stopped_before_openai": True,
            "reason": f"Only {jobs_after_filter} jobs survived the filter (minimum 30 required).",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(json.dumps(results, indent=2))
        return

    capped_jobs = deduplicated_jobs[:max_jobs]
    print(f"Matching the same {len(capped_jobs)} jobs (capped at --max-jobs {max_jobs}), "
          f"{runs} runs at --workers {workers}...")

    model = os.getenv("OPENAI_MODEL")
    per_run_seconds = []
    per_run_prompt_tokens = []
    per_run_completion_tokens = []
    per_run_total_tokens = []
    per_run_cost = []
    per_run_openai_calls = []
    per_run_skipped = []
    per_run_rate_limit_hits = []
    per_run_rate_limit_wait_seconds = []

    for run_number in range(1, runs + 1):
        start = time.perf_counter()
        _matched_jobs, usage_log, stats = _match_with_usage_tracking(
            capped_jobs, profile, max_workers=workers
        )
        elapsed = time.perf_counter() - start

        prompt_tokens = sum(u["prompt_tokens"] for u in usage_log)
        completion_tokens = sum(u["completion_tokens"] for u in usage_log)
        total_tokens = sum(u["total_tokens"] for u in usage_log)
        cost = _estimate_cost_usd(model, prompt_tokens, completion_tokens)

        print(
            f"  run {run_number}/{runs}: {elapsed:.2f}s, {len(usage_log)} calls, "
            f"{stats.get('skipped_jobs', 0)} skipped, "
            f"{stats.get('rate_limit_hits', 0)} 429s, "
            f"{stats.get('rate_limit_wait_seconds', 0.0):.1f}s waited on rate limits"
        )

        per_run_seconds.append(elapsed)
        per_run_prompt_tokens.append(prompt_tokens)
        per_run_completion_tokens.append(completion_tokens)
        per_run_total_tokens.append(total_tokens)
        per_run_cost.append(cost if cost is not None else 0.0)
        per_run_openai_calls.append(len(usage_log))
        per_run_skipped.append(stats.get("skipped_jobs", 0))
        per_run_rate_limit_hits.append(stats.get("rate_limit_hits", 0))
        per_run_rate_limit_wait_seconds.append(stats.get("rate_limit_wait_seconds", 0.0))

    worker_result = {
        "runs": runs,
        "jobs_matched": len(capped_jobs),
        "median_match_seconds": round(_median(per_run_seconds), 3),
        "median_openai_calls": _median(per_run_openai_calls),
        "median_skipped_jobs": _median(per_run_skipped),
        "median_rate_limit_hits": _median(per_run_rate_limit_hits),
        "median_rate_limit_wait_seconds": round(_median(per_run_rate_limit_wait_seconds), 3),
        "median_prompt_tokens": _median(per_run_prompt_tokens),
        "median_completion_tokens": _median(per_run_completion_tokens),
        "median_total_tokens": _median(per_run_total_tokens),
        "median_estimated_cost_usd": round(_median(per_run_cost), 6) if model == PRICED_MODEL else None,
        "all_run_seconds": [round(s, 3) for s in per_run_seconds],
    }

    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {
            "jobs_collected_per_source": jobs_per_source,
            "jobs_collected_total": len(jobs),
            "jobs_after_filter": jobs_after_filter,
            "filter_rejections": filter_rejections,
            "max_jobs_used_for_matching": max_jobs,
            "openai_model": model,
            "by_workers": {},
        }

    results["by_workers"][str(workers)] = worker_result

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {output_path}")
    print(json.dumps(worker_result, indent=2))

    if len(results["by_workers"]) > 1:
        print("\nComparison table (median of each worker count's runs):")
        header = (
            f"  {'workers':<9}{'match_s':<10}{'matched':<9}{'skipped':<9}"
            f"{'429s':<7}{'wait_s':<9}{'tokens':<10}{'cost_usd':<10}"
        )
        print(header)
        for worker_count, data in sorted(results["by_workers"].items(), key=lambda kv: int(kv[0])):
            print(
                f"  {worker_count:<9}{data['median_match_seconds']:<10}"
                f"{data['jobs_matched'] - data['median_skipped_jobs']:<9}{data['median_skipped_jobs']:<9}"
                f"{data['median_rate_limit_hits']:<7}{data['median_rate_limit_wait_seconds']:<9}"
                f"{data['median_total_tokens']:<10}{data['median_estimated_cost_usd']}"
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
    matched_jobs, usage_log, match_stats = _match_with_usage_tracking(deduplicated_jobs, profile)
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
        "skipped_jobs": match_stats.get("skipped_jobs", 0),
        "dropped_matched_skills": match_stats.get("dropped_matched_skills", 0),
        "dropped_missing_skills": match_stats.get("dropped_missing_skills", 0),
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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Run the fair matching-stage comparison mode instead of a full "
            "pipeline run: collect+filter once, then repeat ONLY matching "
            "--runs times at this many max_workers, on the same --max-jobs "
            "jobs. Run this twice (e.g. --workers 1, then --workers 8) with "
            "the same output_name to build a side-by-side comparison."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Comparison mode only: how many times to repeat the matching stage (default: 3).",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=60,
        help="Comparison mode only: cap on jobs matched, taken in stable order after dedup (default: 60).",
    )
    args = parser.parse_args()

    if args.workers is not None:
        run_fair_comparison(
            workers=args.workers, runs=args.runs, max_jobs=args.max_jobs, output_name=args.output_name
        )
        return

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

    rejected_seniority = diagnostics["rejected_seniority_titles"]
    print(f"\nRejected-for-seniority titles -- {sum(rejected_seniority.values())} total:")
    for title, count in rejected_seniority.most_common():
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
