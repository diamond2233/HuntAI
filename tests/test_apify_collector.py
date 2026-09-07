"""Tests for the Apify LinkedIn collector (collectors/apify.py).

No real Apify calls are made -- the ApifyClient is mocked throughout.
"""

from unittest.mock import MagicMock, patch

import pytest
from apify_client.errors import ApifyClientError

from collectors.apify import MAX_SEARCHES, ApifyCollector


def _mock_client(items: list[dict]) -> MagicMock:
    client = MagicMock()
    run = MagicMock()
    run.default_dataset_id = "dataset-1"
    run.status = "SUCCEEDED"
    client.actor.return_value.call.return_value = run
    client.dataset.return_value.list_items.return_value.items = items
    return client


@pytest.fixture(autouse=True)
def apify_token_env(monkeypatch):
    monkeypatch.setenv("APIFY_API_TOKEN", "test-token")


def test_missing_token_raises_clear_runtime_error(monkeypatch):
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)

    with patch("collectors.apify.ApifyClient") as MockClient:
        collector = ApifyCollector(roles=["Software Engineer"], locations=["Bangalore"])
        with pytest.raises(RuntimeError) as exc_info:
            collector.collect()

    assert "APIFY_API_TOKEN" in str(exc_info.value)
    MockClient.assert_not_called()


def test_disabled_apify_never_calls_the_client_when_no_roles_or_locations():
    # ApifyCollector itself has no "enabled" flag (that lives in collect_node
    # config) -- but it must still no-op safely with nothing to search for.
    with patch("collectors.apify.ApifyClient") as MockClient:
        assert ApifyCollector(roles=[], locations=["Bangalore"]).collect() == []
        assert ApifyCollector(roles=["Software Engineer"], locations=[]).collect() == []

    MockClient.assert_not_called()


def test_valid_result_converts_to_job_model():
    raw_result = {
        "jobId": "4458952044",
        "title": "Python Developer",
        "companyName": "Example Company",
        "locationText": "Austin, TX",
        "description": "<p>Build cool stuff.</p>",
        "publicUrl": "https://www.linkedin.com/jobs/view/python-developer-4458952044",
    }
    client = _mock_client([raw_result])

    with patch("collectors.apify.ApifyClient", return_value=client):
        collector = ApifyCollector(roles=["Software Engineer"], locations=["Bangalore"])
        jobs = collector.collect()

    assert len(jobs) == 1
    job = jobs[0]
    assert job.id == "4458952044"
    assert job.title == "Python Developer"
    assert job.company == "Example Company"
    assert job.location == "Austin, TX"
    assert "Build cool stuff" in job.description
    assert "<p>" not in job.description
    assert job.url == "https://www.linkedin.com/jobs/view/python-developer-4458952044"
    assert job.source == "apify"
    assert job.posted_date is None


def test_malformed_result_is_skipped_without_crashing_collection():
    good_result = {
        "jobId": "1",
        "title": "Backend Engineer",
        "companyName": "Acme",
        "publicUrl": "https://linkedin.com/jobs/view/1",
    }
    malformed_result = {"jobId": "2"}  # missing required title/companyName/url

    client = _mock_client([good_result, malformed_result])

    with patch("collectors.apify.ApifyClient", return_value=client):
        collector = ApifyCollector(roles=["Software Engineer"], locations=["Bangalore"])
        jobs = collector.collect()

    assert len(jobs) == 1
    assert jobs[0].id == "1"


def test_searches_are_bounded_regardless_of_location_count():
    client = _mock_client([])
    many_locations = [f"City{i}" for i in range(10)]

    with patch("collectors.apify.ApifyClient", return_value=client):
        collector = ApifyCollector(roles=["Software Engineer"], locations=many_locations)
        collector.collect()

    assert client.actor.return_value.call.call_count == MAX_SEARCHES


def test_roles_are_combined_into_a_single_or_query_per_search():
    client = _mock_client([])

    with patch("collectors.apify.ApifyClient", return_value=client):
        collector = ApifyCollector(
            roles=["Software Engineer", "Backend Engineer"], locations=["Bangalore"]
        )
        collector.collect()

    call_kwargs = client.actor.return_value.call.call_args.kwargs
    assert call_kwargs["run_input"]["keywords"] == "Software Engineer OR Backend Engineer"
    assert call_kwargs["run_input"]["location"] == "Bangalore"


def test_actor_call_failure_raises_clear_runtime_error():
    client = MagicMock()
    client.actor.return_value.call.side_effect = ApifyClientError("boom")

    with patch("collectors.apify.ApifyClient", return_value=client):
        collector = ApifyCollector(roles=["Software Engineer"], locations=["Bangalore"])
        with pytest.raises(RuntimeError) as exc_info:
            collector.collect()

    assert "Bangalore" in str(exc_info.value)


def test_run_with_failed_status_raises_clear_runtime_error():
    """Regression test for a real bug found in production.

    The actor can return a run with a valid default_dataset_id (an empty
    dataset) even when the run itself failed -- e.g. LinkedIn's guest search
    returning "NO_SEARCH_RESULTS". Checking only for a dataset id let 5 real
    failed runs pass silently as "0 jobs found" instead of raising. The
    run's actual status must be checked too.
    """
    client = MagicMock()
    run = MagicMock()
    run.default_dataset_id = "dataset-1"
    run.status = "FAILED"
    run.status_message = "LinkedIn returned no public guest search cards."
    client.actor.return_value.call.return_value = run

    with patch("collectors.apify.ApifyClient", return_value=client):
        collector = ApifyCollector(roles=["Software Engineer"], locations=["Bangalore"])
        with pytest.raises(RuntimeError) as exc_info:
            collector.collect()

    assert "Bangalore" in str(exc_info.value)
    assert "FAILED" in str(exc_info.value)
    client.dataset.assert_not_called()
