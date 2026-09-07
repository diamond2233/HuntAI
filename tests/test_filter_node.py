"""Tests for filter_node's wiring of apply_filters into PipelineState."""

from models.job import Job
from pipeline.graph import filter_node


def _job(title, company="Acme", location="Bangalore, India"):
    return Job(
        id=title,
        title=title,
        company=company,
        location=location,
        url=f"https://example.com/{title}",
        source="greenhouse",
    )


def test_filter_node_stores_result_in_filtered_jobs():
    profile = {
        "excluded_companies": ["TCS"],
        "roles": ["Backend Engineer"],
        "locations": ["Bangalore"],
    }
    jobs = [
        _job("Backend Engineer", company="Acme", location="Bangalore, India"),
        _job("Backend Engineer", company="TCS", location="Bangalore, India"),
        _job("Product Manager", company="Acme", location="Bangalore, India"),
    ]

    result = filter_node({"profile": profile, "jobs": jobs})

    assert len(result["filtered_jobs"]) == 1
    assert result["filtered_jobs"][0].company == "Acme"


def test_filter_node_handles_missing_profile_and_jobs():
    result = filter_node({})
    assert result["filtered_jobs"] == []
