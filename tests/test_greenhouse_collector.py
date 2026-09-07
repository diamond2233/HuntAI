"""Tests for the Greenhouse collector.

No live network access is used -- urllib.request.urlopen is mocked so the
suite stays runnable offline.
"""

import json
from unittest.mock import MagicMock, patch

from collectors.greenhouse import GreenhouseCollector, _html_to_text


def _fake_response(payload: dict) -> MagicMock:
    """Build a mock that behaves like the object urlopen() returns."""
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode("utf-8")
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


def test_valid_greenhouse_job_converts_to_job_model():
    raw_job = {
        "id": 12345,
        "title": "Backend Engineer",
        "absolute_url": "https://boards.greenhouse.io/acme/jobs/12345",
        "location": {"name": "Bangalore, India"},
        "content": "<p>We are looking for a <b>Backend Engineer</b>.</p><p>Apply now!</p>",
        "updated_at": "2024-01-01T10:00:00-05:00",
    }

    with patch("urllib.request.urlopen", return_value=_fake_response({"jobs": [raw_job]})):
        collector = GreenhouseCollector(board_token="acme", company="Acme Corp")
        jobs = collector.collect()

    assert len(jobs) == 1
    job = jobs[0]
    assert job.id == "12345"
    assert job.title == "Backend Engineer"
    assert job.company == "Acme Corp"
    assert job.location == "Bangalore, India"
    assert "Backend Engineer" in job.description
    assert "<p>" not in job.description and "<b>" not in job.description
    assert job.url == "https://boards.greenhouse.io/acme/jobs/12345"
    assert job.source == "greenhouse"
    assert job.posted_date is None


def test_html_description_is_normalized_to_plain_text():
    html = "<p>Great <b>Python</b> role.</p><ul><li>Remote friendly</li></ul>"
    text = _html_to_text(html)

    assert text is not None
    assert "<" not in text
    assert "Python" in text
    assert "Remote friendly" in text


def test_missing_optional_fields_do_not_crash_collector():
    raw_job = {
        "id": 999,
        "title": "Software Engineer",
        "absolute_url": "https://boards.greenhouse.io/acme/jobs/999",
        # no "location" and no "content"
    }

    with patch("urllib.request.urlopen", return_value=_fake_response({"jobs": [raw_job]})):
        collector = GreenhouseCollector(board_token="acme")
        jobs = collector.collect()

    assert len(jobs) == 1
    job = jobs[0]
    assert job.location is None
    assert job.description is None
    assert job.company == "acme"  # falls back to board token when no company given


def test_malformed_job_is_skipped_without_crashing_collection():
    good_job = {
        "id": 1,
        "title": "Data Engineer",
        "absolute_url": "https://boards.greenhouse.io/acme/jobs/1",
    }
    malformed_job = {"id": 2}  # missing required "title" and "absolute_url"

    with patch(
        "urllib.request.urlopen",
        return_value=_fake_response({"jobs": [good_job, malformed_job]}),
    ):
        collector = GreenhouseCollector(board_token="acme")
        jobs = collector.collect()

    assert len(jobs) == 1
    assert jobs[0].id == "1"


def test_http_failure_raises_clear_error():
    import urllib.error

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("boom")):
        collector = GreenhouseCollector(board_token="acme")
        try:
            collector.collect()
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "acme" in str(exc)
