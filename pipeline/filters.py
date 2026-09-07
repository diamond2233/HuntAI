"""Deterministic filtering stage (no LLM).

Removes obvious mismatches before the expensive LLM matching stage ever sees
a job: excluded companies, non-matching job titles, and non-matching
locations. These are reliable, rule-based constraints -- exactly the kind of
work that should NOT cost an LLM call.

Experience and skills are intentionally left out of this stage. See the
comments in apply_filters() for why.
"""

import re
from typing import Optional

from models.job import Job


def is_excluded_company(company: str, excluded_companies: list[str]) -> bool:
    """Case-insensitive check against the excluded_companies list."""
    if not excluded_companies:
        return False
    excluded_lower = {c.strip().lower() for c in excluded_companies}
    return company.strip().lower() in excluded_lower


# Generic technical-role nouns. If a title contains one of these as a whole
# word, it's a plausible engineering/technical role even when it doesn't
# literally match one of the profile's configured role phrases -- e.g.
# "Machine Learning Engineer" or "Platform Engineer" when the profile only
# lists "AI Engineer"/"Backend Engineer". This exists purely to widen
# deterministic recall; it makes no judgment about skill or seniority fit --
# that's still the LLM matcher's job.
GENERIC_TECHNICAL_TITLE_WORDS = ["engineer", "developer"]

# Titles containing one of these as a whole word read as management/
# leadership roles, not individual-contributor technical roles, and are
# rejected even if they also contain a word like "Engineer" (e.g. "Engineer
# Manager"). Checked before the generic technical words above, so this takes
# priority.
NON_TECHNICAL_TITLE_WORDS = ["manager", "director", "head", "vp", "vice president", "chief", "president"]


def _contains_whole_phrase(text: str, phrase: str) -> bool:
    """Case-insensitive, whole-word/whole-phrase substring check.

    Plain substring matching would let "Software Engineering Manager" match
    the phrase "Software Engineer", since "Software Engineer" is literally a
    prefix of "Software Engineering". Regex word boundaries (\\b) prevent
    that: the phrase must appear as whole words, not as a fragment glued to
    a longer word.
    """
    pattern = r"\b" + re.escape(phrase.strip()) + r"\b"
    return re.search(pattern, text, re.IGNORECASE) is not None


def matches_role(title: str, roles: list[str]) -> bool:
    """Decide whether a title is worth sending on to the LLM matcher.

    Two tiers, in order:
      1. Exact match against one of the profile's configured role phrases
         (whole-word/whole-phrase, case-insensitive).
      2. A recall net: generic technical titles (containing "Engineer" or
         "Developer") are also accepted, unless the title reads as a
         management/leadership role -- even though they don't literally
         match a configured role phrase. This keeps recall reasonable
         without turning the filter into a semantic judge; the LLM matcher
         still decides actual suitability.

    No roles configured means we don't filter by title at all.
    """
    if not roles:
        return True

    if any(_contains_whole_phrase(title, role) for role in roles):
        return True

    if any(_contains_whole_phrase(title, word) for word in NON_TECHNICAL_TITLE_WORDS):
        return False

    return any(_contains_whole_phrase(title, word) for word in GENERIC_TECHNICAL_TITLE_WORDS)


def matches_location(location: Optional[str], locations: list[str]) -> bool:
    """Case-insensitive substring match against configured locations.

    No locations configured means we don't filter by location at all. But if
    locations ARE configured, a job with no location string can't be
    confirmed to match, so it's rejected.
    """
    if not locations:
        return True
    if not location:
        return False

    location_lower = location.lower()
    return any(loc.strip().lower() in location_lower for loc in locations)


def apply_filters(jobs: list[Job], profile: dict) -> list[Job]:
    """Apply the deterministic filter rules to a list of jobs.

    Filters applied here: excluded company, role/title, location.

    NOT filtered here:
      - Experience: Job has no structured experience field, and reliably
        extracting years-of-experience requirements from free-text job
        descriptions isn't a deterministic problem -- it needs judgment,
        which belongs to the later LLM matching stage.
      - Skills: skill relevance is semantic (e.g. "GenAI" vs "Generative AI"
        vs "RAG" overlap, or a skill implied without being named), which
        keyword matching can't judge reliably. That's left to the LLM
        matcher too.
    """
    excluded_companies = profile.get("excluded_companies") or []
    roles = profile.get("roles") or []
    locations = profile.get("locations") or []

    filtered_jobs = []
    for job in jobs:
        if is_excluded_company(job.company, excluded_companies):
            continue
        if not matches_role(job.title, roles):
            continue
        if not matches_location(job.location, locations):
            continue
        filtered_jobs.append(job)

    return filtered_jobs
