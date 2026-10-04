"""Skipped-jobs report exporter.

Writes a report of every job the matching stage skipped (failed after
retries), with enough detail to say exactly why -- so a skip is never
silent. See pipeline/matcher.py's MatchFailure for where these details
come from.
"""

import json
from pathlib import Path


def export_skipped_report(
    skipped_jobs: int, skipped_details: list[dict], output_path: str | Path
) -> None:
    """Write a skipped-jobs report, even when nothing was skipped.

    Always writing the file (with an empty list when skipped_jobs is 0)
    means a reader never has to wonder whether a missing file means "no
    skips" or "this run didn't get far enough to check".
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "skipped_jobs": skipped_jobs,
        "skipped_details": skipped_details,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
