"""JSON exporter.

Writes the complete ranked job match information to a JSON file, preserving
full JobMatch/Job structure rather than flattening it.
"""

import json
from pathlib import Path

from models.job import JobMatch


def export_json(ranked_jobs: list[JobMatch], output_path: str | Path) -> None:
    """Write ranked_jobs to a JSON file, in order, with full detail intact."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = [
        {
            "rank": rank,
            "job": match.job.model_dump(mode="json"),
            "score": match.score,
            "matched_skills": match.matched_skills,
            "missing_skills": match.missing_skills,
            "rationale": match.rationale,
        }
        for rank, match in enumerate(ranked_jobs, start=1)
    ]

    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
