"""Lever job board collector.

Fetches job postings from a public Lever job board and converts them into our
normalized Job model.

HTTP client choice: same reasoning as GreenhouseCollector -- Lever's public
postings API is a single, unauthenticated GET request that returns JSON, so
Python's standard library (urllib.request) is used instead of adding a
third-party HTTP library.
"""

import json
import logging
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from typing import Optional

from collectors.base import BaseCollector
from collectors.html_utils import html_to_text
from models.job import Job

logger = logging.getLogger(__name__)

# Lever's public Postings API. {company} is the slug visible in a company's
# careers URL (jobs.lever.co/<company>). mode=json returns full posting
# content (including HTML description) in the same response.
LEVER_POSTINGS_URL = "https://api.lever.co/v0/postings/{company}?mode=json"

REQUEST_TIMEOUT_SECONDS = 10


def _normalize_location(categories: Optional[dict]) -> Optional[str]:
    """Lever nests location under categories.location; it may be absent."""
    if not categories:
        return None
    location = categories.get("location")
    if location and location.strip():
        return location.strip()
    return None


def _extract_description(raw_posting: dict) -> Optional[str]:
    """Extract a posting's description from wherever Lever actually put it.

    Lever's live API puts "description"/"descriptionPlain" directly on the
    posting object (siblings of "id"/"text"), NOT nested under a "content"
    object as older documentation suggested. This checks the top-level
    fields first (where real data has been observed) and falls back to a
    nested "content" object if present, so this keeps working even if an
    account/API version does nest it.

    Lever also carries substantive requirements in "lists" (e.g. "What
    you'll do", "You should apply if") as separate structured sections. That
    content comes straight from the source too, so it's appended -- leaving
    it out was a real cause of descriptions looking empty/thin even when the
    posting had a "description" field with only a short company blurb.
    """
    content = raw_posting.get("content")
    if not isinstance(content, dict):
        content = {}

    description_html = (
        raw_posting.get("descriptionHtml")
        or raw_posting.get("description")
        or content.get("descriptionHtml")
        or content.get("description")
    )

    parts = []
    main_text = html_to_text(description_html)
    if main_text:
        parts.append(main_text)

    lists = raw_posting.get("lists")
    if not isinstance(lists, list):
        lists = content.get("lists") if isinstance(content.get("lists"), list) else []

    for item in lists:
        if not isinstance(item, dict):
            continue
        section_text = html_to_text(item.get("content"))
        if not section_text:
            continue
        heading = (item.get("text") or "").strip()
        parts.append(f"{heading}: {section_text}" if heading else section_text)

    return "\n\n".join(parts) if parts else None


def _parse_posted_date(created_at_ms: object) -> Optional[date]:
    """Convert Lever's createdAt (epoch milliseconds) into a date.

    Unlike Greenhouse's public API, Lever's createdAt explicitly represents
    when the posting was created in Lever, so it's a reasonable posted_date --
    not just an arbitrary "last updated" timestamp.
    """
    if not isinstance(created_at_ms, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(created_at_ms / 1000, tz=timezone.utc).date()
    except (OverflowError, OSError, ValueError):
        return None


class LeverCollector(BaseCollector):
    """Collects job postings from a single company's Lever job board.

    The company slug is passed in (not hardcoded) so this collector works for
    any company's Lever board. `company` is an optional display name for the
    Job.company field; if omitted, the slug itself is used.
    """

    def __init__(self, company_slug: str, company: Optional[str] = None):
        self.company_slug = company_slug
        self.company = company or company_slug

    def collect(self) -> list[Job]:
        raw_postings = self._fetch_postings()

        jobs: list[Job] = []
        for raw_posting in raw_postings:
            try:
                jobs.append(self._to_job(raw_posting))
            except (KeyError, ValueError, TypeError) as exc:
                # A single malformed posting should not take down the whole
                # collection run -- skip it and keep going.
                logger.warning(
                    "Skipping malformed Lever posting (company=%s, id=%s): %s",
                    self.company_slug,
                    raw_posting.get("id", "unknown") if isinstance(raw_posting, dict) else "unknown",
                    exc,
                )
        return jobs

    def _fetch_postings(self) -> list[dict]:
        url = LEVER_POSTINGS_URL.format(company=self.company_slug)

        try:
            with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                raw_body = response.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Failed to fetch postings from Lever company '{self.company_slug}': {exc}"
            ) from exc

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Lever company '{self.company_slug}' returned invalid JSON: {exc}"
            ) from exc

        # Unlike Greenhouse (which wraps jobs in {"jobs": [...]}), Lever's
        # postings endpoint returns a bare JSON array at the top level.
        if not isinstance(payload, list):
            raise RuntimeError(
                f"Unexpected response structure from Lever company "
                f"'{self.company_slug}': expected a JSON list."
            )
        return payload

    def _to_job(self, raw_posting: dict) -> Job:
        # Required fields. Missing any of these means the record is unusable;
        # letting the KeyError propagate lets collect() skip just this posting.
        posting_id = raw_posting["id"]
        title = raw_posting["text"]
        url = raw_posting.get("hostedUrl") or raw_posting["applyUrl"]

        return Job(
            id=str(posting_id),
            title=title,
            company=self.company,
            location=_normalize_location(raw_posting.get("categories")),
            description=_extract_description(raw_posting),
            url=url,
            source="lever",
            posted_date=_parse_posted_date(raw_posting.get("createdAt")),
        )
