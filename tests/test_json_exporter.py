"""Tests for the JSON exporter (exporters/json.py)."""

import json

from exporters.json import export_json
from models.job import Job, JobMatch


def _match(job_id, score, title="Backend Engineer", company="Acme", matched=None, missing=None):
    job = Job(
        id=job_id,
        title=title,
        company=company,
        location="Bangalore",
        description="A job description.",
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


def test_file_is_created(tmp_path):
    output_path = tmp_path / "jobs.json"
    export_json([_match("1", 90)], output_path)
    assert output_path.exists()


def test_output_is_valid_json(tmp_path):
    output_path = tmp_path / "jobs.json"
    export_json([_match("1", 90)], output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert isinstance(data, list)


def test_rank_and_score_are_present(tmp_path):
    output_path = tmp_path / "jobs.json"
    export_json([_match("1", 90), _match("2", 70)], output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data[0]["rank"] == 1
    assert data[0]["score"] == 90
    assert data[1]["rank"] == 2
    assert data[1]["score"] == 70


def test_nested_job_information_is_preserved(tmp_path):
    output_path = tmp_path / "jobs.json"
    export_json(
        [_match("1", 90, title="Backend Software Engineer", company="Example Inc")], output_path
    )

    data = json.loads(output_path.read_text(encoding="utf-8"))
    job = data[0]["job"]
    assert job["id"] == "1"
    assert job["title"] == "Backend Software Engineer"
    assert job["company"] == "Example Inc"
    assert job["location"] == "Bangalore"
    assert job["description"] == "A job description."
    assert job["url"] == "https://example.com/1"
    assert job["source"] == "greenhouse"
    assert job["posted_date"] is None


def test_matched_missing_and_rationale_are_preserved(tmp_path):
    output_path = tmp_path / "jobs.json"
    export_json([_match("1", 90, matched=["Python"], missing=["Go"])], output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data[0]["matched_skills"] == ["Python"]
    assert data[0]["missing_skills"] == ["Go"]
    assert data[0]["rationale"] == "Strong overlap."


def test_multiple_jobs_preserve_ranking_order(tmp_path):
    output_path = tmp_path / "jobs.json"
    matches = [_match("1", 90, title="First"), _match("2", 80, title="Second")]
    export_json(matches, output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert [item["job"]["title"] for item in data] == ["First", "Second"]


def test_empty_input_produces_valid_empty_json_list(tmp_path):
    output_path = tmp_path / "jobs.json"
    export_json([], output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data == []
