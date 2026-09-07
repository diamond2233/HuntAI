"""Tests for the deduplication stage (pipeline/dedup.py).

Pure function tests -- no network, no LLM.
"""

from models.job import Job
from pipeline.dedup import deduplicate, normalize_url


def _job(job_id: str, url: str, title: str = "Software Engineer") -> Job:
    return Job(
        id=job_id,
        title=title,
        company="Acme",
        url=url,
        source="greenhouse",
    )


def test_duplicate_urls_are_removed():
    jobs = [_job("1", "https://example.com/jobs/123"), _job("2", "https://example.com/jobs/123")]
    result = deduplicate(jobs)
    assert len(result) == 1


def test_first_occurrence_is_preserved():
    jobs = [
        _job("1", "https://example.com/jobs/123", title="First Seen"),
        _job("2", "https://example.com/jobs/123", title="Duplicate"),
    ]
    result = deduplicate(jobs)
    assert result[0].title == "First Seen"


def test_order_is_preserved():
    job_a = _job("A", "https://example.com/a")
    job_b = _job("B", "https://example.com/b")
    job_a_dup = _job("A-dup", "https://example.com/a")
    job_c = _job("C", "https://example.com/c")
    job_b_dup = _job("B-dup", "https://example.com/b")

    result = deduplicate([job_a, job_b, job_a_dup, job_c, job_b_dup])

    assert [job.id for job in result] == ["A", "B", "C"]


def test_different_urls_remain_separate():
    jobs = [_job("1", "https://example.com/1"), _job("2", "https://example.com/2")]
    result = deduplicate(jobs)
    assert len(result) == 2


def test_url_fragments_are_ignored_for_deduplication():
    jobs = [
        _job("1", "https://example.com/jobs/123"),
        _job("2", "https://example.com/jobs/123#apply"),
    ]
    result = deduplicate(jobs)
    assert len(result) == 1


def test_leading_and_trailing_whitespace_is_normalized():
    assert normalize_url("  https://example.com/1  ") == "https://example.com/1"
    jobs = [
        _job("1", "https://example.com/1"),
        _job("2", "  https://example.com/1  "),
    ]
    result = deduplicate(jobs)
    assert len(result) == 1


def test_blank_url_does_not_crash():
    jobs = [_job("1", "")]
    result = deduplicate(jobs)
    assert len(result) == 1


def test_two_blank_urls_are_not_treated_as_duplicates():
    jobs = [_job("1", ""), _job("2", "")]
    result = deduplicate(jobs)
    assert len(result) == 2


def test_empty_input_returns_empty_list():
    assert deduplicate([]) == []
