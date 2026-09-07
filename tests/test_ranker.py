"""Tests for the ranking stage (pipeline/ranker.py).

Pure function tests -- no network, no LLM.
"""

from models.job import Job, JobMatch
from pipeline.ranker import rank_jobs


def _match(job_id: str, score: float) -> JobMatch:
    job = Job(
        id=job_id,
        title="Software Engineer",
        company="Acme",
        url=f"https://example.com/{job_id}",
        source="greenhouse",
    )
    return JobMatch(job=job, score=score, rationale="Because reasons.")


def test_jobs_are_sorted_highest_score_first():
    matches = [_match("1", 40), _match("2", 90), _match("3", 60)]
    result = rank_jobs(matches)
    assert [m.job.id for m in result] == ["2", "3", "1"]


def test_lowest_score_appears_last():
    matches = [_match("1", 40), _match("2", 90), _match("3", 60)]
    result = rank_jobs(matches)
    assert result[-1].job.id == "1"


def test_equal_scores_preserve_original_order():
    matches = [_match("1", 70), _match("2", 70), _match("3", 70)]
    result = rank_jobs(matches)
    assert [m.job.id for m in result] == ["1", "2", "3"]


def test_equal_scores_mixed_with_different_scores_preserve_relative_order():
    matches = [_match("1", 70), _match("2", 90), _match("3", 70), _match("4", 90)]
    result = rank_jobs(matches)
    assert [m.job.id for m in result] == ["2", "4", "1", "3"]


def test_different_scores_are_correctly_ordered():
    matches = [_match("1", 10), _match("2", 55), _match("3", 99), _match("4", 30)]
    result = rank_jobs(matches)
    assert [m.score for m in result] == [99, 55, 30, 10]


def test_empty_input_returns_empty_list():
    assert rank_jobs([]) == []


def test_original_list_is_not_modified():
    matches = [_match("1", 40), _match("2", 90)]
    original_order = list(matches)

    rank_jobs(matches)

    assert matches == original_order


def test_original_list_object_is_not_the_returned_list():
    matches = [_match("1", 40), _match("2", 90)]
    result = rank_jobs(matches)
    assert result is not matches


def test_all_job_match_objects_remain_intact_after_ranking():
    matches = [_match("1", 40), _match("2", 90)]
    result = rank_jobs(matches)

    assert len(result) == len(matches)
    for match in matches:
        assert match in result
    for match in result:
        assert isinstance(match, JobMatch)
        assert isinstance(match.job, Job)
