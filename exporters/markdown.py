"""Markdown exporter.

Writes ranked job matches to a human-readable Markdown file. This module
owns all Markdown formatting -- callers (pipeline/graph.py) just decide when
to call it and where to write.
"""

from pathlib import Path

from models.job import JobMatch


def _format_skills(skills: list[str]) -> str:
    return ", ".join(skills) if skills else "None"


def export_markdown(ranked_jobs: list[JobMatch], output_path: str | Path) -> None:
    """Write ranked_jobs to a Markdown file, one section per job, in order."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# HuntAI Job Matches", ""]

    if not ranked_jobs:
        lines.append("No matching jobs found.")
    else:
        for rank, match in enumerate(ranked_jobs, start=1):
            job = match.job
            lines.append(f"## {rank}. {job.title} — {job.company}")
            lines.append("")
            lines.append(f"**Score:** {match.score}/100")
            lines.append(f"**Location:** {job.location or 'Not specified'}")
            lines.append(f"**Source:** {job.source}")
            lines.append(f"**URL:** {job.url}")
            lines.append("")
            lines.append(f"**Matched Skills:** {_format_skills(match.matched_skills)}")
            lines.append(f"**Missing Skills:** {_format_skills(match.missing_skills)}")
            lines.append("")
            lines.append(f"**Why it matches:** {match.rationale}")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
