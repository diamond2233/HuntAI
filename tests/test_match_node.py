"""Tests for match_node's wiring of match_jobs into PipelineState."""

from unittest.mock import patch

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

    with patch("pipeline.graph.match_jobs", return_value=expected_matches) as mock_match_jobs:
        result = match_node({"profile": profile, "filtered_jobs": jobs})

    mock_match_jobs.assert_called_once_with(jobs, profile)
    assert result["matched_jobs"] == expected_matches


def test_match_node_handles_missing_jobs_and_profile():
    with patch("pipeline.graph.match_jobs", return_value=[]) as mock_match_jobs:
        result = match_node({})

    mock_match_jobs.assert_called_once_with([], {})
    assert result["matched_jobs"] == []
