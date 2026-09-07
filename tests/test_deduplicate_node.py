"""Tests for deduplicate_node's wiring of deduplicate() into PipelineState."""

from models.job import Job
from pipeline.graph import deduplicate_node


def _job(job_id: str, url: str) -> Job:
    return Job(
        id=job_id,
        title="Software Engineer",
        company="Acme",
        url=url,
        source="greenhouse",
    )


def test_deduplicate_node_updates_filtered_jobs():
    jobs = [
        _job("1", "https://example.com/1"),
        _job("2", "https://example.com/1#apply"),
        _job("3", "https://example.com/2"),
    ]

    result = deduplicate_node({"filtered_jobs": jobs})

    assert [job.id for job in result["filtered_jobs"]] == ["1", "3"]


def test_deduplicate_node_handles_missing_filtered_jobs():
    result = deduplicate_node({})
    assert result["filtered_jobs"] == []
