"""Tests for the Markdown exporter (exporters/markdown.py)."""

from exporters.markdown import export_markdown
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
        rationale="Strong overlap with required skills.",
    )


def test_file_is_created(tmp_path):
    output_path = tmp_path / "jobs.md"
    export_markdown([_match("1", 90)], output_path)
    assert output_path.exists()


def test_job_details_appear(tmp_path):
    output_path = tmp_path / "jobs.md"
    export_markdown(
        [_match("1", 95, title="Backend Software Engineer", company="Example Inc")], output_path
    )

    content = output_path.read_text(encoding="utf-8")
    assert "Backend Software Engineer" in content
    assert "Example Inc" in content
    assert "95" in content


def test_rank_is_correct_for_multiple_jobs(tmp_path):
    output_path = tmp_path / "jobs.md"
    matches = [_match("1", 90, title="Job One"), _match("2", 70, title="Job Two")]
    export_markdown(matches, output_path)

    content = output_path.read_text(encoding="utf-8")
    assert "1. Job One" in content
    assert "2. Job Two" in content
    assert content.index("1. Job One") < content.index("2. Job Two")


def test_matched_and_missing_skills_appear(tmp_path):
    output_path = tmp_path / "jobs.md"
    export_markdown([_match("1", 80, matched=["Python", "SQL"], missing=["Go"])], output_path)

    content = output_path.read_text(encoding="utf-8")
    assert "Python, SQL" in content
    assert "Go" in content


def test_empty_input_creates_valid_markdown_file(tmp_path):
    output_path = tmp_path / "jobs.md"
    export_markdown([], output_path)

    content = output_path.read_text(encoding="utf-8")
    assert "# HuntAI Job Matches" in content
    assert "No matching jobs found." in content


def test_multiple_jobs_preserve_ranking_order(tmp_path):
    output_path = tmp_path / "jobs.md"
    matches = [
        _match("1", 90, title="First"),
        _match("2", 80, title="Second"),
        _match("3", 70, title="Third"),
    ]
    export_markdown(matches, output_path)

    content = output_path.read_text(encoding="utf-8")
    assert content.index("First") < content.index("Second") < content.index("Third")


def test_output_is_utf8(tmp_path):
    output_path = tmp_path / "jobs.md"
    export_markdown([_match("1", 90, company="Café Corp")], output_path)

    decoded = output_path.read_bytes().decode("utf-8")
    assert "Café Corp" in decoded


def test_creates_parent_directory_if_missing(tmp_path):
    output_path = tmp_path / "nested" / "dir" / "jobs.md"
    export_markdown([_match("1", 90)], output_path)
    assert output_path.exists()
