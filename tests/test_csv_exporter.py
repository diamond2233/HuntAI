"""Tests for the CSV exporter (exporters/csv.py)."""

import csv

from exporters.csv import export_csv
from models.job import Job, JobMatch


def _match(job_id, score, title="Backend Engineer", company="Acme", matched=None, missing=None):
    job = Job(
        id=job_id,
        title=title,
        company=company,
        location="Bangalore",
        url=f"https://example.com/{job_id}",
        source="greenhouse",
    )
    return JobMatch(
        job=job,
        score=score,
        matched_skills=matched or [],
        missing_skills=missing or [],
        rationale="Strong overlap.",
    )


def _read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_header(path):
    with open(path, newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def test_file_is_created(tmp_path):
    output_path = tmp_path / "jobs.csv"
    export_csv([_match("1", 90)], output_path)
    assert output_path.exists()


def test_header_contains_required_columns(tmp_path):
    output_path = tmp_path / "jobs.csv"
    export_csv([_match("1", 90)], output_path)

    header = _read_header(output_path)
    assert header == [
        "rank",
        "score",
        "title",
        "company",
        "location",
        "source",
        "url",
        "matched_skills",
        "missing_skills",
        "rationale",
    ]


def test_one_row_per_job(tmp_path):
    output_path = tmp_path / "jobs.csv"
    matches = [_match("1", 90), _match("2", 80), _match("3", 70)]
    export_csv(matches, output_path)

    rows = _read_rows(output_path)
    assert len(rows) == 3


def test_rank_and_score_are_correct(tmp_path):
    output_path = tmp_path / "jobs.csv"
    matches = [_match("1", 90), _match("2", 80)]
    export_csv(matches, output_path)

    rows = _read_rows(output_path)
    assert rows[0]["rank"] == "1"
    assert float(rows[0]["score"]) == 90
    assert rows[1]["rank"] == "2"
    assert float(rows[1]["score"]) == 80


def test_skills_are_serialized_correctly(tmp_path):
    output_path = tmp_path / "jobs.csv"
    export_csv([_match("1", 90, matched=["Python", "SQL"], missing=[])], output_path)

    rows = _read_rows(output_path)
    assert rows[0]["matched_skills"] == "Python, SQL"
    assert rows[0]["missing_skills"] == "None"


def test_multiple_jobs_preserve_ranking_order(tmp_path):
    output_path = tmp_path / "jobs.csv"
    matches = [_match("1", 90, title="First"), _match("2", 80, title="Second")]
    export_csv(matches, output_path)

    rows = _read_rows(output_path)
    assert rows[0]["title"] == "First"
    assert rows[1]["title"] == "Second"


def test_empty_input_creates_header_row_only(tmp_path):
    output_path = tmp_path / "jobs.csv"
    export_csv([], output_path)

    rows = _read_rows(output_path)
    assert rows == []
    assert _read_header(output_path)[0] == "rank"
