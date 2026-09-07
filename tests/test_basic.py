"""Basic tests proving the core Pydantic models work correctly."""

import pytest
from pydantic import ValidationError

from models.job import Job, JobMatch


def test_job_can_be_instantiated():
    job = Job(
        id="gh-123",
        title="Backend Engineer",
        company="Acme Corp",
        location="Remote",
        description="Build backend services.",
        url="https://boards.greenhouse.io/acme/jobs/123",
        source="greenhouse",
    )

    assert job.title == "Backend Engineer"
    assert job.company == "Acme Corp"
    assert job.posted_date is None


def test_job_missing_required_field_raises_validation_error():
    with pytest.raises(ValidationError):
        Job(title="Backend Engineer", company="Acme Corp")


def test_job_match_can_be_instantiated():
    job = Job(
        id="lever-456",
        title="Software Engineer",
        company="Example Inc",
        url="https://jobs.lever.co/example/456",
        source="lever",
    )

    match = JobMatch(
        job=job,
        score=0.85,
        matched_skills=["Python", "SQL"],
        missing_skills=["Go"],
        rationale="Strong overlap with required backend skills.",
    )

    assert match.score == 0.85
    assert match.job.title == "Software Engineer"
