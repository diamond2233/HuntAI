"""Probes public Greenhouse and Lever job-board APIs to find new boards worth
adding to config/profile.yaml.

Run manually with:
    python scripts/check_boards.py

For each company in COMPANIES, generates a few slug guesses (lowercase, no
spaces, hyphenated, first word only) plus any known ALIASES, and checks both:
    Greenhouse: https://boards-api.greenhouse.io/v1/boards/{slug}/jobs
    Lever:      https://api.lever.co/v0/postings/{slug}?mode=json

A handful of extra slugs (EXTRA_GREENHOUSE_SLUGS / EXTRA_LEVER_SLUGS) are
checked directly, one platform each, without generating variants.

This script only reads from the public internet and prints a report -- it
never writes to config/profile.yaml or creates a collector. Deciding what to
actually add is a separate, manual step based on what this reports.

Because a slug like "meta" or "apple" could easily belong to an unrelated
company, every hit is marked VERIFIED or UNVERIFIED based on whether the
board's own name plausibly matches the company we were guessing for.
"""

import json
import re
import sys
import time
import urllib.error
import urllib.request

# Some real job titles contain characters the Windows console's default
# codepage can't encode (e.g. en-dashes, accents). Replace rather than crash.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REQUEST_TIMEOUT_SECONDS = 8
DELAY_BETWEEN_REQUESTS_SECONDS = 0.5

GREENHOUSE_JOBS_URL = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
GREENHOUSE_BOARD_INFO_URL = "https://boards-api.greenhouse.io/v1/boards/{slug}"
LEVER_POSTINGS_URL = "https://api.lever.co/v0/postings/{slug}?mode=json"

COMPANIES = [
    "google", "microsoft", "atlassian", "jp morgan", "morgan stanley",
    "goldman sachs", "de shaw", "juspay", "eternal", "palo alto", "adobe",
    "american express", "amazon", "flipkart", "apple", "arcesium", "coindcx",
    "meta", "netflix", "hotstar", "phonepe", "uber", "intuit", "servicenow",
    "wells fargo", "walmart", "zerodha", "upstox", "udaan",
    "texas instruments", "swiggy", "salesforce", "qualcomm", "nvidia",
    "paypal", "ola", "myntra", "linkedin", "ibm", "dream11", "airbus",
]

# company (lowercase) -> extra slug guesses, for cases a mechanical
# transformation of the name wouldn't find (rebrands, known short names).
ALIASES = {
    "eternal": ["zomato"],
    "hotstar": ["jiostar"],
    "goldman sachs": ["goldmansachs", "gs"],
    "de shaw": ["deshaw"],
    "american express": ["americanexpress", "amex"],
    "palo alto": ["paloaltonetworks"],
    "texas instruments": ["texasinstruments"],
    "jp morgan": ["jpmorgan", "jpmorganchase"],
    "dream11": ["dreamsports"],
    "ola": ["olacabs"],
    "upstox": ["rksv"],
}

EXTRA_GREENHOUSE_SLUGS = [
    "stripe", "databricks", "postman", "cloudflare", "mongodb", "twilio",
    "elastic", "coinbase", "okta", "datadog", "gitlab", "airbnb",
    "pinterest", "dropbox", "samsara",
]
EXTRA_LEVER_SLUGS = ["spotify", "palantir", "plaid"]

ENGINEER_TITLE_PATTERN = re.compile(r"\b(engineer|developer)\b", re.IGNORECASE)
INDIA_OR_REMOTE_PATTERN = re.compile(r"\b(india|remote)\b", re.IGNORECASE)
INDIA_PATTERN = re.compile(r"\bindia\b", re.IGNORECASE)
NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9]")


def _slug_variants(company: str) -> list[tuple[str, str]]:
    """Generate (slug, verification_hint) guesses for a company name.

    verification_hint is what we'd expect the real board's name to contain
    -- normally the company name itself, but for an alias (e.g. "eternal"
    guessing the slug "zomato") it's the alias, since that's the name the
    real board will actually show.
    """
    variants: list[tuple[str, str]] = []
    seen_slugs: set[str] = set()

    def add(slug: str, hint: str) -> None:
        if slug and slug not in seen_slugs:
            seen_slugs.add(slug)
            variants.append((slug, hint))

    lower = company.lower().strip()
    add(lower.replace(" ", ""), company)
    add(lower.replace(" ", "-"), company)
    words = lower.split()
    if len(words) > 1:
        add(words[0], company)

    for alias in ALIASES.get(lower, []):
        add(alias, alias)

    return variants


def _get_json(url: str) -> tuple[int, object | None]:
    """GET a URL, returning (status_code, parsed_json_or_None).

    Never raises: a 404, timeout, connection error, or invalid JSON all just
    come back as a miss (status 0 covers anything that isn't a clean HTTP
    response at all).
    """
    try:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            status = response.status
            body = response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, None

    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, None


def _squash(text: str) -> str:
    """Lowercase, letters/digits only -- so punctuation/spacing/casing
    differences ("J.P. Morgan" vs "jpmorgan") don't cause a false UNVERIFIED.
    """
    return NON_ALPHANUMERIC_PATTERN.sub("", text.lower())


def _name_plausibly_matches(board_name: str, hint: str) -> bool:
    squashed_name = _squash(board_name)
    squashed_hint = _squash(hint)
    if not squashed_name or not squashed_hint:
        return False
    return squashed_hint in squashed_name or squashed_name in squashed_hint


def _looks_like_india_or_remote(location: str | None) -> bool:
    return bool(location) and INDIA_OR_REMOTE_PATTERN.search(location) is not None


def _looks_like_explicitly_india(location: str | None) -> bool:
    return bool(location) and INDIA_PATTERN.search(location) is not None


def _looks_like_engineering_title(title: str) -> bool:
    return bool(title) and ENGINEER_TITLE_PATTERN.search(title) is not None


def _summarize_jobs(jobs: list[dict], get_location: "callable", get_title: "callable") -> dict:
    """Two relevance counts, because they disagree a lot in practice:

    - india_or_remote_jobs: the loose count (location contains "india" OR
      "remote"). This is what was literally asked for, but for a global
      company it mostly just means "Remote - US"/"Remote - Canada"/etc,
      which is NOT relevant to an India-based job search.
    - explicit_india_jobs: location contains "india" specifically. This is
      the stricter, honest number actually used to decide what's worth
      adding to profile.yaml.
    """
    india_or_remote_jobs = [job for job in jobs if _looks_like_india_or_remote(get_location(job))]
    explicit_india_jobs = [job for job in jobs if _looks_like_explicitly_india(get_location(job))]
    return {
        "india_or_remote_jobs": len(india_or_remote_jobs),
        "engineering_jobs_india_or_remote": sum(
            1 for job in india_or_remote_jobs if _looks_like_engineering_title(get_title(job))
        ),
        "explicit_india_jobs": len(explicit_india_jobs),
        "engineering_jobs_explicit_india": sum(
            1 for job in explicit_india_jobs if _looks_like_engineering_title(get_title(job))
        ),
    }


def _check_greenhouse(company: str, slug: str, hint: str) -> dict | None:
    time.sleep(DELAY_BETWEEN_REQUESTS_SECONDS)
    status, jobs_payload = _get_json(GREENHOUSE_JOBS_URL.format(slug=slug))
    if status != 200 or not isinstance(jobs_payload, dict):
        return None
    jobs = jobs_payload.get("jobs")
    if not isinstance(jobs, list):
        return None

    time.sleep(DELAY_BETWEEN_REQUESTS_SECONDS)
    _, board_info = _get_json(GREENHOUSE_BOARD_INFO_URL.format(slug=slug))
    board_name = board_info.get("name") if isinstance(board_info, dict) else None

    summary = _summarize_jobs(
        jobs,
        get_location=lambda job: (job.get("location") or {}).get("name"),
        get_title=lambda job: job.get("title") or "",
    )

    return {
        "company": company,
        "platform": "greenhouse",
        "slug": slug,
        "board_name": board_name,
        "total_jobs": len(jobs),
        **summary,
        "sample_titles": [job.get("title") for job in jobs[:3]],
        "verified": bool(board_name) and _name_plausibly_matches(board_name, hint),
    }


def _check_lever(company: str, slug: str, hint: str) -> dict | None:
    time.sleep(DELAY_BETWEEN_REQUESTS_SECONDS)
    status, postings = _get_json(LEVER_POSTINGS_URL.format(slug=slug))
    if status != 200 or not isinstance(postings, list):
        return None

    board_name = None
    if postings:
        hosted_url = postings[0].get("hostedUrl") or ""
        match = re.search(r"jobs\.lever\.co/([^/]+)/", hosted_url)
        board_name = match.group(1) if match else None

    summary = _summarize_jobs(
        postings,
        get_location=lambda job: (job.get("categories") or {}).get("location"),
        get_title=lambda job: job.get("text") or "",
    )

    return {
        "company": company,
        "platform": "lever",
        "slug": slug,
        "board_name": board_name,
        "total_jobs": len(postings),
        **summary,
        "sample_titles": [job.get("text") for job in postings[:3]],
        "verified": bool(board_name) and _name_plausibly_matches(board_name, hint),
    }


def _print_hit(result: dict) -> None:
    status = "VERIFIED" if result["verified"] else "UNVERIFIED"
    print(
        f"  HIT [{status}] {result['company']} | {result['platform']} | slug={result['slug']!r} | "
        f"board_name={result['board_name']!r} | total={result['total_jobs']} | "
        f"india_or_remote={result['india_or_remote_jobs']} (engineering={result['engineering_jobs_india_or_remote']}) | "
        f"explicit_india={result['explicit_india_jobs']} (engineering={result['engineering_jobs_explicit_india']})"
    )
    for title in result["sample_titles"]:
        print(f"      sample: {title}")


def probe_all() -> list[dict]:
    hits: list[dict] = []
    checked_companies = set()

    for company in COMPANIES:
        checked_companies.add(company)
        print(f"Checking '{company}'...")
        for slug, hint in _slug_variants(company):
            gh_result = _check_greenhouse(company, slug, hint)
            if gh_result:
                hits.append(gh_result)
                _print_hit(gh_result)

            lever_result = _check_lever(company, slug, hint)
            if lever_result:
                hits.append(lever_result)
                _print_hit(lever_result)

    print("\nChecking extra Greenhouse-only slugs...")
    for slug in EXTRA_GREENHOUSE_SLUGS:
        company = slug.capitalize()
        gh_result = _check_greenhouse(company, slug, slug)
        if gh_result:
            hits.append(gh_result)
            _print_hit(gh_result)

    print("\nChecking extra Lever-only slugs...")
    for slug in EXTRA_LEVER_SLUGS:
        company = slug.capitalize()
        lever_result = _check_lever(company, slug, slug)
        if lever_result:
            hits.append(lever_result)
            _print_hit(lever_result)

    # A "miss" is a company from COMPANIES with no hit at all, on either
    # platform, under any slug variant we tried.
    companies_with_hits = {hit["company"] for hit in hits}
    misses = sorted(checked_companies - companies_with_hits)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    verified_hits = [hit for hit in hits if hit["verified"]]
    unverified_hits = [hit for hit in hits if not hit["verified"]]

    print(f"\nVerified hits ({len(verified_hits)}):")
    print("  (explicit_india = location contains 'India'; this is the number that matters for Step 2 --")
    print("   india_or_remote is the loose literal metric, shown for comparison, mostly non-India noise)")
    for hit in verified_hits:
        print(
            f"  {hit['company']:<20} {hit['platform']:<10} slug={hit['slug']:<20} "
            f"total={hit['total_jobs']:<5} india_or_remote={hit['india_or_remote_jobs']:<5} "
            f"(eng={hit['engineering_jobs_india_or_remote']:<4}) explicit_india={hit['explicit_india_jobs']:<5} "
            f"(eng={hit['engineering_jobs_explicit_india']})"
        )

    print(f"\nUnverified hits ({len(unverified_hits)}):")
    for hit in unverified_hits:
        print(
            f"  {hit['company']:<20} {hit['platform']:<10} slug={hit['slug']:<20} "
            f"board_name={hit['board_name']!r:<25} total={hit['total_jobs']}"
        )

    print(f"\nMisses ({len(misses)} of {len(checked_companies)} companies, no hit on either platform):")
    for company in misses:
        print(f"  {company}")

    return hits


if __name__ == "__main__":
    probe_all()
