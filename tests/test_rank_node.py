"""Tests for rank_node's wiring of rank_jobs into PipelineState."""

from models.job import Job, JobMatch
from pipeline.graph import rank_node


def _match(job_id: str, score: float) -> JobMatch:
    job = Job(
        id=job_id,
        title="Software Engineer",
        company="Acme",
        url=f"https://example.com/{job_id}",
        source="greenhouse",
    )
    return JobMatch(job=job, score=score, rationale="Because reasons.")


def test_rank_node_stores_sorted_result_in_ranked_jobs():
    matched_jobs = [_match("1", 40), _match("2", 90), _match("3", 60)]

    result = rank_node({"matched_jobs": matched_jobs})

    assert [m.job.id for m in result["ranked_jobs"]] == ["2", "3", "1"]


def test_rank_node_handles_missing_matched_jobs():
    result = rank_node({})
    assert result["ranked_jobs"] == []
