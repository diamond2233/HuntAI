"""Tests for match_node's wiring of match_jobs into PipelineState.

match_jobs is mocked throughout. Since it's mocked, it never populates the
`stats` dict match_node passes it -- tests that care about skipped-jobs
behavior simulate that by using `side_effect` to mutate `stats` themselves,
the same way the real match_jobs would.
"""

from pathlib import Path
from unittest.mock import ANY, patch

from models.job import Job, JobMatch
from pipeline.graph import match_node


def _job(job_id: str) -> Job:
    return Job(
        id=job_id,
        title="Software Engineer",
        company="Acme",
        url=f"https://example.com/{job_id}",
        source="greenhouse",
    )


def test_match_node_stores_result_in_matched_jobs():
    profile = {"roles": ["Software Engineer"]}
    jobs = [_job("1")]
    expected_matches = [JobMatch(job=jobs[0], score=80, rationale="Good fit.")]

    with (
        patch("pipeline.graph.match_jobs", return_value=expected_matches) as mock_match_jobs,
        patch("pipeline.graph.export_skipped_report"),
    ):
        result = match_node({"profile": profile, "filtered_jobs": jobs})

    mock_match_jobs.assert_called_once_with(jobs, profile, stats=ANY)
    assert result["matched_jobs"] == expected_matches


def test_match_node_handles_missing_jobs_and_profile():
    with (
        patch("pipeline.graph.match_jobs", return_value=[]) as mock_match_jobs,
        patch("pipeline.graph.export_skipped_report"),
    ):
        result = match_node({})

    mock_match_jobs.assert_called_once_with([], {}, stats=ANY)
    assert result["matched_jobs"] == []


def _match_jobs_with_skips(skipped_details):
    """Build a match_jobs replacement that populates `stats` the way the
    real one would, then returns an empty match list."""

    def fake_match_jobs(jobs, profile, stats=None, max_workers=None):
        if stats is not None:
            stats["skipped_jobs"] = len(skipped_details)
            stats["skipped_details"] = skipped_details
        return []

    return fake_match_jobs


def test_match_node_returns_zero_skipped_jobs_when_nothing_skipped():
    with (
        patch("pipeline.graph.match_jobs", side_effect=_match_jobs_with_skips([])),
        patch("pipeline.graph.export_skipped_report") as mock_export,
    ):
        result = match_node({"profile": {}, "filtered_jobs": [_job("1")]})

    assert result["skipped_jobs"] == 0
    assert result["skipped_details"] == []
    mock_export.assert_called_once_with(0, [], Path("output/skipped.json"))


def test_match_node_returns_skipped_jobs_and_details():
    details = [
        {
            "id": "1",
            "title": "Backend Engineer",
            "company": "Acme",
            "error_type": "RateLimitError",
            "error_message": "Rate limit reached.",
            "attempts": 4,
        }
    ]

    with (
        patch("pipeline.graph.match_jobs", side_effect=_match_jobs_with_skips(details)),
        patch("pipeline.graph.export_skipped_report") as mock_export,
    ):
        result = match_node({"profile": {}, "filtered_jobs": [_job("1")]})

    assert result["skipped_jobs"] == 1
    assert result["skipped_details"] == details
    mock_export.assert_called_once_with(1, details, Path("output/skipped.json"))


def test_match_node_prints_warning_when_jobs_are_skipped(capsys):
    details = [
        {"id": "1", "title": "Job A", "company": "Acme", "error_type": "Refusal", "error_message": "m", "attempts": 1}
    ]

    with (
        patch("pipeline.graph.match_jobs", side_effect=_match_jobs_with_skips(details)),
        patch("pipeline.graph.export_skipped_report"),
    ):
        match_node({"profile": {}, "filtered_jobs": [_job("1"), _job("2")]})

    captured = capsys.readouterr()
    assert "WARNING: 1 of 2 jobs skipped" in captured.out
    assert "output" in captured.out and "skipped.json" in captured.out


def test_match_node_prints_no_warning_when_nothing_is_skipped(capsys):
    with (
        patch("pipeline.graph.match_jobs", side_effect=_match_jobs_with_skips([])),
        patch("pipeline.graph.export_skipped_report"),
    ):
        match_node({"profile": {}, "filtered_jobs": [_job("1")]})

    captured = capsys.readouterr()
    assert "WARNING" not in captured.out
