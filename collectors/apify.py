"""Apify LinkedIn job search collector.

Uses the `jobsapi/linkedin-jobs-search-scraper` Apify Actor (search mode) to
find LinkedIn postings and converts them into our normalized Job model.

Unlike GreenhouseCollector/LeverCollector (one call per configured board/
company), Apify search is keyword+location based, so a naive "one call per
role per location" loop could multiply out of control. To keep this bounded,
ApifyCollector runs at most MAX_SEARCHES actor calls -- one per configured
location (extra locations beyond that are simply not searched), with all
configured roles combined into a single "OR" keyword query per call rather
than one call per role.
"""

import logging
import os

from apify_client import ApifyClient
from apify_client.errors import ApifyClientError
from dotenv import load_dotenv

from collectors.base import BaseCollector
from collectors.html_utils import html_to_text
from models.job import Job

load_dotenv()

logger = logging.getLogger(__name__)

APIFY_ACTOR_ID = "jobsapi/linkedin-jobs-search-scraper"

# Bounds how many actor runs a single collect() call can make, regardless of
# how many locations are configured -- this is what keeps the implementation
# practical rather than one search per (role, location) combination.
MAX_SEARCHES = 5

DEFAULT_MAX_ITEMS = 100
DEFAULT_MAX_PAGES = 2


class ApifyCollector(BaseCollector):
    """Searches LinkedIn (via Apify) for the configured roles and locations.

    roles/locations are passed in (not read from config here) so this
    collector stays a plain "talk to the source" component -- deciding what
    to search for is collect_node's orchestration job, same as it decides
    which Greenhouse boards/Lever companies to run.
    """

    def __init__(
        self,
        roles: list[str],
        locations: list[str],
        max_items: int = DEFAULT_MAX_ITEMS,
        max_pages: int = DEFAULT_MAX_PAGES,
    ):
        self.roles = roles
        self.locations = locations
        self.max_items = max_items
        self.max_pages = max_pages

    def collect(self) -> list[Job]:
        if not self.roles or not self.locations:
            return []

        api_token = os.getenv("APIFY_API_TOKEN")
        if not api_token:
            raise RuntimeError(
                "APIFY_API_TOKEN is not set. Add it to .env before running the Apify collector."
            )

        client = ApifyClient(token=api_token)
        keywords = " OR ".join(self.roles)

        jobs: list[Job] = []
        for location in self.locations[:MAX_SEARCHES]:
            raw_results = self._search(client, keywords, location)
            for raw_result in raw_results:
                try:
                    jobs.append(self._to_job(raw_result))
                except (KeyError, ValueError, TypeError) as exc:
                    # A single malformed result should not take down the
                    # whole collection run -- skip it and keep going.
                    logger.warning(
                        "Skipping malformed Apify result (location=%s, id=%s): %s",
                        location,
                        raw_result.get("jobId", "unknown") if isinstance(raw_result, dict) else "unknown",
                        exc,
                    )
        return jobs

    def _search(self, client: ApifyClient, keywords: str, location: str) -> list[dict]:
        run_input = {
            "mode": "search",
            "keywords": keywords,
            "location": location,
            "maxItems": self.max_items,
            "maxPages": self.max_pages,
        }

        try:
            run = client.actor(APIFY_ACTOR_ID).call(run_input=run_input)
        except ApifyClientError as exc:
            raise RuntimeError(
                f"Apify actor call failed for location '{location}': {exc}"
            ) from exc

        if not run or not run.default_dataset_id:
            raise RuntimeError(
                f"Apify actor run for location '{location}' did not return a dataset."
            )

        if run.status != "SUCCEEDED":
            raise RuntimeError(
                f"Apify actor run for location '{location}' did not succeed "
                f"(status={run.status}, message={run.status_message})."
            )

        try:
            return client.dataset(run.default_dataset_id).list_items().items
        except ApifyClientError as exc:
            raise RuntimeError(
                f"Failed to fetch Apify results for location '{location}': {exc}"
            ) from exc

    def _to_job(self, raw_result: dict) -> Job:
        # Required fields. Missing any of these means the record is unusable;
        # letting the KeyError propagate lets collect() skip just this result.
        job_id = raw_result["jobId"]
        title = raw_result["title"]
        company = raw_result["companyName"]
        url = raw_result.get("publicUrl") or raw_result["url"]

        return Job(
            id=str(job_id),
            title=title,
            company=company,
            location=raw_result.get("locationText"),
            description=html_to_text(raw_result.get("description")),
            url=url,
            source="apify",
            # The actor's output doesn't include a verified original posting
            # date, so this is left None rather than guessed.
            posted_date=None,
        )
