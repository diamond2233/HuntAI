"""Tests for export_node's orchestration of the three exporters.

The exporter functions are mocked -- these tests never write real files.
"""

from pathlib import Path
from unittest.mock import patch

from models.job import Job, JobMatch
from pipeline.graph import export_node


def _match(job_id: str) -> JobMatch:
    job = Job(
        id=job_id,
        title="Software Engineer",
        company="Acme",
        url=f"https://example.com/{job_id}",
        source="greenhouse",
    )
    return JobMatch(job=job, score=80, rationale="Good fit.")


def test_export_node_reads_ranked_jobs_and_calls_all_exporters():
    ranked_jobs = [_match("1")]

    with (
        patch("pipeline.graph.export_markdown") as mock_markdown,
        patch("pipeline.graph.export_csv") as mock_csv,
        patch("pipeline.graph.export_json") as mock_json,
    ):
        result = export_node({"ranked_jobs": ranked_jobs})

    mock_markdown.assert_called_once()
    mock_csv.assert_called_once()
    mock_json.assert_called_once()
    assert result["ranked_jobs"] == ranked_jobs


def test_export_node_uses_expected_output_paths():
    ranked_jobs = [_match("1")]

    with (
        patch("pipeline.graph.export_markdown") as mock_markdown,
        patch("pipeline.graph.export_csv") as mock_csv,
        patch("pipeline.graph.export_json") as mock_json,
    ):
        export_node({"ranked_jobs": ranked_jobs})

    assert Path(mock_markdown.call_args.args[1]) == Path("output/jobs.md")
    assert Path(mock_csv.call_args.args[1]) == Path("output/jobs.csv")
    assert Path(mock_json.call_args.args[1]) == Path("output/jobs.json")


def test_export_node_passes_ranked_jobs_to_each_exporter():
    ranked_jobs = [_match("1"), _match("2")]

    with (
        patch("pipeline.graph.export_markdown") as mock_markdown,
        patch("pipeline.graph.export_csv") as mock_csv,
        patch("pipeline.graph.export_json") as mock_json,
    ):
        export_node({"ranked_jobs": ranked_jobs})

    assert mock_markdown.call_args.args[0] == ranked_jobs
    assert mock_csv.call_args.args[0] == ranked_jobs
    assert mock_json.call_args.args[0] == ranked_jobs


def test_export_node_handles_missing_ranked_jobs():
    with (
        patch("pipeline.graph.export_markdown") as mock_markdown,
        patch("pipeline.graph.export_csv") as mock_csv,
        patch("pipeline.graph.export_json") as mock_json,
    ):
        result = export_node({})

    mock_markdown.assert_called_once_with([], Path("output/jobs.md"))
    mock_csv.assert_called_once_with([], Path("output/jobs.csv"))
    mock_json.assert_called_once_with([], Path("output/jobs.json"))
    assert result["ranked_jobs"] == []
