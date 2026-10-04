"""Tests for the skipped-jobs report exporter (exporters/skipped.py)."""

import json

from exporters.skipped import export_skipped_report


def test_file_is_created(tmp_path):
    output_path = tmp_path / "skipped.json"
    export_skipped_report(0, [], output_path)
    assert output_path.exists()


def test_file_is_created_even_with_no_skips(tmp_path):
    # A missing file must never be the only signal that nothing was
    # skipped -- the file always exists after a run.
    output_path = tmp_path / "skipped.json"
    export_skipped_report(0, [], output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data == {"skipped_jobs": 0, "skipped_details": []}


def test_skipped_count_and_details_are_written(tmp_path):
    output_path = tmp_path / "skipped.json"
    details = [
        {
            "id": "1",
            "title": "Backend Engineer",
            "company": "Acme",
            "error_type": "RateLimitError",
            "error_message": "Rate limit reached.",
            "attempts": 4,
        }
    ]
    export_skipped_report(1, details, output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["skipped_jobs"] == 1
    assert data["skipped_details"] == details


def test_multiple_details_preserve_order(tmp_path):
    output_path = tmp_path / "skipped.json"
    details = [
        {"id": "1", "title": "Job A", "company": "Acme", "error_type": "Refusal", "error_message": "m", "attempts": 1},
        {"id": "2", "title": "Job B", "company": "Acme", "error_type": "OpenAIError", "error_message": "m", "attempts": 1},
    ]
    export_skipped_report(2, details, output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert [d["title"] for d in data["skipped_details"]] == ["Job A", "Job B"]


def test_creates_parent_directory_if_missing(tmp_path):
    output_path = tmp_path / "nested" / "dir" / "skipped.json"
    export_skipped_report(0, [], output_path)
    assert output_path.exists()
