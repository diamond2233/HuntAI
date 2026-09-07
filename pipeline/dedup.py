"""Deduplication stage (deterministic, no LLM, no fuzzy/semantic matching).

We remove duplicate postings before LLM matching so that we don't spend
multiple LLM calls on the same job, and so the final results never contain
duplicate recommendations.

The only question this stage answers is: "Have we already seen this job
posting?" -- identity is based purely on the job URL.
"""

from urllib.parse import urlsplit, urlunsplit

from models.job import Job


def normalize_url(url: str) -> str:
    """Normalize a URL just enough to catch trivial duplicates.

    Strips surrounding whitespace and drops the fragment (e.g. "#apply"),
    since a fragment identifies a spot on the page, not a different job.
    Scheme, host, path, and query string are left untouched -- we don't try
    to guess which query params are tracking-only, so none are removed.
    """
    url = url.strip()
    if not url:
        return url

    scheme, netloc, path, query, _fragment = urlsplit(url)
    return urlunsplit((scheme, netloc, path, query, ""))


def deduplicate(jobs: list[Job]) -> list[Job]:
    """Remove duplicate job postings, keyed by normalized URL.

    Keeps the first occurrence of each URL and preserves input order.
    """
    seen_keys: set[str] = set()
    unique_jobs: list[Job] = []

    for index, job in enumerate(jobs):
        normalized_url = normalize_url(job.url)

        if normalized_url:
            key = normalized_url
        else:
            # A blank URL can't identify duplicates, so treat this job as
            # unique by its position rather than merging all blank-URL jobs
            # into a single entry.
            key = f"__blank_url_{index}__"

        if key not in seen_keys:
            seen_keys.add(key)
            unique_jobs.append(job)

    return unique_jobs
