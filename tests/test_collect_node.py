"""Tests for collect_node's orchestration of configured collectors.

GreenhouseCollector and LeverCollector are both mocked -- these tests never
touch a real API. They only verify that collect_node reads state["profile"]
correctly and wires jobs into state["jobs"].
"""

from unittest.mock import MagicMock, patch

from models.job import Job
from pipeline.graph import collect_node


def _make_job(job_id: str, source: str = "greenhouse") -> Job:
    return Job(
        id=job_id,
        title="Engineer",
        company="Acme",
        url=f"https://example.com/{job_id}",
        source=source,
    )


def test_configured_greenhouse_board_instantiates_and_calls_collector():
    profile = {
        "sources": {
            "greenhouse": {
                "enabled": True,
                "boards": [{"token": "acme", "company": "Acme Corp"}],
            }
        }
    }
    expected_jobs = [_make_job("1"), _make_job("2")]

    with patch("pipeline.graph.GreenhouseCollector") as MockCollector:
        MockCollector.return_value.collect.return_value = expected_jobs

        result = collect_node({"profile": profile, "jobs": []})

    MockCollector.assert_called_once_with(board_token="acme", company="Acme Corp")
    assert result["jobs"] == expected_jobs


def test_configured_lever_company_instantiates_and_calls_collector():
    profile = {
        "sources": {
            "lever": {
                "enabled": True,
                "companies": [{"slug": "acme", "company": "Acme Corp"}],
            }
        }
    }
    expected_jobs = [_make_job("1", source="lever")]

    with patch("pipeline.graph.LeverCollector") as MockCollector:
        MockCollector.return_value.collect.return_value = expected_jobs

        result = collect_node({"profile": profile, "jobs": []})

    MockCollector.assert_called_once_with(company_slug="acme", company="Acme Corp")
    assert result["jobs"] == expected_jobs


def test_greenhouse_and_lever_jobs_are_combined():
    profile = {
        "sources": {
            "greenhouse": {
                "enabled": True,
                "boards": [{"token": "acme", "company": "Acme Corp"}],
            },
            "lever": {
                "enabled": True,
                "companies": [{"slug": "beta", "company": "Beta Inc"}],
            },
        }
    }

    with (
        patch("pipeline.graph.GreenhouseCollector") as MockGreenhouse,
        patch("pipeline.graph.LeverCollector") as MockLever,
    ):
        MockGreenhouse.return_value.collect.return_value = [_make_job("gh-1", "greenhouse")]
        MockLever.return_value.collect.return_value = [_make_job("lv-1", "lever")]

        result = collect_node({"profile": profile, "jobs": []})

    assert {job.id for job in result["jobs"]} == {"gh-1", "lv-1"}


def test_multiple_greenhouse_boards_have_jobs_combined():
    profile = {
        "sources": {
            "greenhouse": {
                "enabled": True,
                "boards": [
                    {"token": "acme", "company": "Acme Corp"},
                    {"token": "beta", "company": "Beta Inc"},
                ],
            }
        }
    }

    def fake_collector(board_token, company=None):
        instance = MagicMock()
        instance.collect.return_value = [_make_job(board_token)]
        return instance

    with patch("pipeline.graph.GreenhouseCollector", side_effect=fake_collector):
        result = collect_node({"profile": profile, "jobs": []})

    assert {job.id for job in result["jobs"]} == {"acme", "beta"}


def test_multiple_lever_companies_have_jobs_combined():
    profile = {
        "sources": {
            "lever": {
                "enabled": True,
                "companies": [
                    {"slug": "acme", "company": "Acme Corp"},
                    {"slug": "beta", "company": "Beta Inc"},
                ],
            }
        }
    }

    def fake_collector(company_slug, company=None):
        instance = MagicMock()
        instance.collect.return_value = [_make_job(company_slug, source="lever")]
        return instance

    with patch("pipeline.graph.LeverCollector", side_effect=fake_collector):
        result = collect_node({"profile": profile, "jobs": []})

    assert {job.id for job in result["jobs"]} == {"acme", "beta"}


def test_disabled_sources_are_not_called():
    profile = {
        "sources": {
            "greenhouse": {"enabled": False, "boards": [{"token": "acme"}]},
            "lever": {"enabled": False, "companies": [{"slug": "acme"}]},
        }
    }

    with (
        patch("pipeline.graph.GreenhouseCollector") as MockGreenhouse,
        patch("pipeline.graph.LeverCollector") as MockLever,
    ):
        result = collect_node({"profile": profile, "jobs": []})

    MockGreenhouse.assert_not_called()
    MockLever.assert_not_called()
    assert result["jobs"] == []


def test_missing_sources_config_does_not_crash():
    with (
        patch("pipeline.graph.GreenhouseCollector") as MockGreenhouse,
        patch("pipeline.graph.LeverCollector") as MockLever,
    ):
        result = collect_node({"profile": {}, "jobs": []})

    MockGreenhouse.assert_not_called()
    MockLever.assert_not_called()
    assert result["jobs"] == []


def test_empty_boards_and_companies_lists_collect_nothing():
    profile = {
        "sources": {
            "greenhouse": {"enabled": True, "boards": []},
            "lever": {"enabled": True, "companies": []},
        }
    }

    with (
        patch("pipeline.graph.GreenhouseCollector") as MockGreenhouse,
        patch("pipeline.graph.LeverCollector") as MockLever,
    ):
        result = collect_node({"profile": profile, "jobs": []})

    MockGreenhouse.assert_not_called()
    MockLever.assert_not_called()
    assert result["jobs"] == []


def test_configured_apify_source_instantiates_and_calls_collector():
    profile = {
        "roles": ["Software Engineer", "Backend Engineer"],
        "locations": ["Bangalore", "Remote"],
        "sources": {
            "apify": {"enabled": True, "max_items": 50, "max_pages": 1},
        },
    }
    expected_jobs = [_make_job("li-1", source="apify")]

    with patch("pipeline.graph.ApifyCollector") as MockCollector:
        MockCollector.return_value.collect.return_value = expected_jobs

        result = collect_node({"profile": profile, "jobs": []})

    MockCollector.assert_called_once_with(
        roles=["Software Engineer", "Backend Engineer"],
        locations=["Bangalore", "Remote"],
        max_items=50,
        max_pages=1,
    )
    assert result["jobs"] == expected_jobs


def test_disabled_apify_is_not_called():
    profile = {
        "roles": ["Software Engineer"],
        "locations": ["Bangalore"],
        "sources": {"apify": {"enabled": False}},
    }

    with patch("pipeline.graph.ApifyCollector") as MockCollector:
        result = collect_node({"profile": profile, "jobs": []})

    MockCollector.assert_not_called()
    assert result["jobs"] == []


def test_all_three_sources_are_combined():
    profile = {
        "roles": ["Software Engineer"],
        "locations": ["Bangalore"],
        "sources": {
            "greenhouse": {"enabled": True, "boards": [{"token": "acme"}]},
            "lever": {"enabled": True, "companies": [{"slug": "beta"}]},
            "apify": {"enabled": True},
        },
    }

    with (
        patch("pipeline.graph.GreenhouseCollector") as MockGreenhouse,
        patch("pipeline.graph.LeverCollector") as MockLever,
        patch("pipeline.graph.ApifyCollector") as MockApify,
    ):
        MockGreenhouse.return_value.collect.return_value = [_make_job("gh-1", "greenhouse")]
        MockLever.return_value.collect.return_value = [_make_job("lv-1", "lever")]
        MockApify.return_value.collect.return_value = [_make_job("ap-1", "apify")]

        result = collect_node({"profile": profile, "jobs": []})

    assert {job.id for job in result["jobs"]} == {"gh-1", "lv-1", "ap-1"}
