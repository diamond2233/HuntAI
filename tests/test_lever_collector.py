"""Tests for the Lever collector.

No live network access is used -- urllib.request.urlopen is mocked so the
suite stays runnable offline.
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from collectors.lever import LeverCollector


def _fake_response(payload) -> MagicMock:
    """Build a mock that behaves like the object urlopen() returns."""
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode("utf-8")
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


def test_valid_lever_posting_converts_to_job_model():
    raw_posting = {
        "id": "abc123",
        "text": "Site Reliability Engineer",
        "hostedUrl": "https://jobs.lever.co/acme/abc123",
        "applyUrl": "https://jobs.lever.co/acme/abc123/apply",
        "categories": {"location": "Remote", "team": "Infra"},
        "content": {"descriptionHtml": "<p>Join our <b>SRE</b> team.</p>"},
        "createdAt": 1700000000000,  # epoch milliseconds
    }

    with patch("urllib.request.urlopen", return_value=_fake_response([raw_posting])):
        collector = LeverCollector(company_slug="acme", company="Acme Corp")
        jobs = collector.collect()

    assert len(jobs) == 1
    job = jobs[0]
    assert job.id == "abc123"
    assert job.title == "Site Reliability Engineer"
    assert job.company == "Acme Corp"
    assert job.location == "Remote"
    assert "SRE" in job.description
    assert "<p>" not in job.description and "<b>" not in job.description
    assert job.url == "https://jobs.lever.co/acme/abc123"
    assert job.source == "lever"
    expected_date = datetime.fromtimestamp(1700000000000 / 1000, tz=timezone.utc).date()
    assert job.posted_date == expected_date


def test_missing_optional_fields_do_not_crash_collector():
    raw_posting = {
        "id": "xyz",
        "text": "Backend Engineer",
        "applyUrl": "https://jobs.lever.co/acme/xyz/apply",
        # no "hostedUrl", "categories", "content", or "createdAt"
    }

    with patch("urllib.request.urlopen", return_value=_fake_response([raw_posting])):
        collector = LeverCollector(company_slug="acme")
        jobs = collector.collect()

    assert len(jobs) == 1
    job = jobs[0]
    assert job.url == "https://jobs.lever.co/acme/xyz/apply"
    assert job.location is None
    assert job.description is None
    assert job.posted_date is None
    assert job.company == "acme"  # falls back to slug when no company given


def test_multiple_postings_are_converted():
    postings = [
        {"id": "1", "text": "Engineer A", "hostedUrl": "https://jobs.lever.co/acme/1"},
        {"id": "2", "text": "Engineer B", "hostedUrl": "https://jobs.lever.co/acme/2"},
    ]

    with patch("urllib.request.urlopen", return_value=_fake_response(postings)):
        collector = LeverCollector(company_slug="acme")
        jobs = collector.collect()

    assert {job.id for job in jobs} == {"1", "2"}


def test_malformed_posting_is_skipped_without_crashing_collection():
    good_posting = {"id": "1", "text": "Engineer A", "hostedUrl": "https://jobs.lever.co/acme/1"}
    malformed_posting = {"id": "2"}  # missing required "text" and any URL

    with patch(
        "urllib.request.urlopen",
        return_value=_fake_response([good_posting, malformed_posting]),
    ):
        collector = LeverCollector(company_slug="acme")
        jobs = collector.collect()

    assert len(jobs) == 1
    assert jobs[0].id == "1"


def test_description_is_extracted_from_top_level_fields():
    """Regression test for the real bug found in production.

    Lever's live API puts "description"/"descriptionPlain" directly on the
    posting object -- NOT nested under a "content" object. The collector
    used to only check "content", so every real Lever posting ended up with
    description=None. This reproduces the actual shape observed from a real
    API response.
    """
    raw_posting = {
        "id": "abc123",
        "text": "Data Scientist",
        "hostedUrl": "https://jobs.lever.co/acme/abc123",
        "description": "<div><p>We build fraud detection models.</p></div>",
        "descriptionPlain": "We build fraud detection models.",
    }

    with patch("urllib.request.urlopen", return_value=_fake_response([raw_posting])):
        collector = LeverCollector(company_slug="acme")
        jobs = collector.collect()

    assert jobs[0].description is not None
    assert "fraud detection" in jobs[0].description
    assert "<p>" not in jobs[0].description


def test_description_includes_lists_sections():
    """Lever's "lists" sections (e.g. "What you'll do") carry real
    requirements content and are appended to the description."""
    raw_posting = {
        "id": "abc123",
        "text": "Data Scientist",
        "hostedUrl": "https://jobs.lever.co/acme/abc123",
        "description": "<p>Short company blurb.</p>",
        "lists": [
            {"text": "What you'll do", "content": "<ul><li>Build models</li></ul>"},
            {"text": "You should apply if", "content": "<ul><li>You know Python</li></ul>"},
        ],
    }

    with patch("urllib.request.urlopen", return_value=_fake_response([raw_posting])):
        collector = LeverCollector(company_slug="acme")
        jobs = collector.collect()

    description = jobs[0].description
    assert "Short company blurb" in description
    assert "Build models" in description
    assert "You know Python" in description
    assert "What you'll do" in description


def test_description_falls_back_to_nested_content_object():
    """If an account/API version does nest fields under "content", that
    still works -- this is a fallback, not the primary path."""
    raw_posting = {
        "id": "abc123",
        "text": "Data Scientist",
        "hostedUrl": "https://jobs.lever.co/acme/abc123",
        "content": {
            "description": "<p>Nested description.</p>",
            "lists": [{"text": "Requirements", "content": "<ul><li>SQL</li></ul>"}],
        },
    }

    with patch("urllib.request.urlopen", return_value=_fake_response([raw_posting])):
        collector = LeverCollector(company_slug="acme")
        jobs = collector.collect()

    description = jobs[0].description
    assert "Nested description" in description
    assert "SQL" in description


def test_no_description_or_lists_present_results_in_none():
    raw_posting = {
        "id": "abc123",
        "text": "Data Scientist",
        "hostedUrl": "https://jobs.lever.co/acme/abc123",
    }

    with patch("urllib.request.urlopen", return_value=_fake_response([raw_posting])):
        collector = LeverCollector(company_slug="acme")
        jobs = collector.collect()

    assert jobs[0].description is None


def test_http_failure_raises_clear_error():
    import urllib.error

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("boom")):
        collector = LeverCollector(company_slug="acme")
        try:
            collector.collect()
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "acme" in str(exc)
