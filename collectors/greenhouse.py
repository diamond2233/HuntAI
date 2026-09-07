"""Greenhouse job board collector.

Fetches jobs from a public Greenhouse job board and converts them into our
normalized Job model.

HTTP client choice: Greenhouse's public job-board API is a single,
unauthenticated GET request that returns JSON. That does not justify a third
-party HTTP library, so this collector uses Python's standard library
(urllib.request) instead of adding `requests` to requirements.txt.
"""

import json
import logging
import urllib.error
import urllib.request
from typing import Optional

from collectors.base import BaseCollector
from collectors.html_utils import html_to_text as _html_to_text
from models.job import Job

logger = logging.getLogger(__name__)

# Greenhouse's public Job Board API. {board_token} identifies the company's
# board (visible in their careers URL, e.g. boards.greenhouse.io/<board_token>).
# `content=true` asks Greenhouse to include the full HTML job description in
# the same response, avoiding a second request per job.
GREENHOUSE_JOBS_URL = "https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs?content=true"

REQUEST_TIMEOUT_SECONDS = 10


def _normalize_location(raw_location: Optional[dict]) -> Optional[str]:
    """Greenhouse represents location as {"name": "..."}; it may be absent."""
    if not raw_location:
        return None
    name = raw_location.get("name")
    if not name or not name.strip():
        return None
    return name.strip()


class GreenhouseCollector(BaseCollector):
    """Collects jobs from a single Greenhouse job board.

    The board token is passed in (not hardcoded) so this collector works for
    any company's Greenhouse board. `company` is an optional display name for
    the Job.company field; if omitted, the board token itself is used.
    """

    def __init__(self, board_token: str, company: Optional[str] = None):
        self.board_token = board_token
        self.company = company or board_token

    def collect(self) -> list[Job]:
        raw_jobs = self._fetch_jobs()

        jobs: list[Job] = []
        for raw_job in raw_jobs:
            try:
                jobs.append(self._to_job(raw_job))
            except (KeyError, ValueError, TypeError) as exc:
                # A single malformed job record should not take down the
                # whole collection run -- skip it and keep going.
                logger.warning(
                    "Skipping malformed Greenhouse job (board=%s, id=%s): %s",
                    self.board_token,
                    raw_job.get("id", "unknown") if isinstance(raw_job, dict) else "unknown",
                    exc,
                )
        return jobs

    def _fetch_jobs(self) -> list[dict]:
        url = GREENHOUSE_JOBS_URL.format(board_token=self.board_token)

        try:
            with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                raw_body = response.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Failed to fetch jobs from Greenhouse board '{self.board_token}': {exc}"
            ) from exc

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Greenhouse board '{self.board_token}' returned invalid JSON: {exc}"
            ) from exc

        jobs = payload.get("jobs") if isinstance(payload, dict) else None
        if not isinstance(jobs, list):
            raise RuntimeError(
                f"Unexpected response structure from Greenhouse board "
                f"'{self.board_token}': expected a 'jobs' list."
            )
        return jobs

    def _to_job(self, raw_job: dict) -> Job:
        # Required fields. Missing any of these means the record is unusable;
        # letting the KeyError propagate lets collect() skip just this job.
        job_id = raw_job["id"]
        title = raw_job["title"]
        url = raw_job["absolute_url"]

        return Job(
            id=str(job_id),
            title=title,
            company=self.company,
            location=_normalize_location(raw_job.get("location")),
            description=_html_to_text(raw_job.get("content")),
            url=url,
            source="greenhouse",
            # Greenhouse's public job board API only exposes `updated_at`
            # (last modified time), not the original posting date. Treating
            # `updated_at` as posted_date would misrepresent when the job was
            # actually posted, so we leave it None until a reliable source
            # of the true posting date is available.
            posted_date=None,
        )
