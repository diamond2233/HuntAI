"""CSV exporter.

Writes ranked job matches to a CSV file, one row per job, using only the
standard library's csv module.
"""

import csv
from pathlib import Path

from models.job import JobMatch

FIELDNAMES = [
    "rank",
    "score",
    "title",
    "company",
    "location",
    "source",
    "url",
    "matched_skills",
    "missing_skills",
    "rationale",
]


def _format_skills(skills: list[str]) -> str:
    return ", ".join(skills) if skills else "None"


def export_csv(ranked_jobs: list[JobMatch], output_path: str | Path) -> None:
    """Write ranked_jobs to a CSV file with one row per job, in order."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # newline="" is required by the csv module to avoid extra blank lines.
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for rank, match in enumerate(ranked_jobs, start=1):
            job = match.job
            writer.writerow(
                {
                    "rank": rank,
                    "score": match.score,
                    "title": job.title,
                    "company": job.company,
                    "location": job.location or "",
                    "source": job.source,
                    "url": job.url,
                    "matched_skills": _format_skills(match.matched_skills),
                    "missing_skills": _format_skills(match.missing_skills),
                    "rationale": match.rationale,
                }
            )
