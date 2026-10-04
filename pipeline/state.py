"""Shared state passed between nodes in the LangGraph pipeline.

LangGraph nodes read from and write to a single shared state object. We use a
TypedDict, which is the standard way LangGraph expects state to be defined.

Each field represents the data available at a given stage of the pipeline:

    profile         -> the loaded user profile (config/profile.yaml)
    jobs            -> raw jobs collected from all sources
    filtered_jobs   -> jobs remaining after hard filtering
    matched_jobs    -> jobs with LLM-produced JobMatch results
    ranked_jobs     -> matched jobs sorted by final rank
    skipped_jobs    -> count of jobs that failed matching after retries
    skipped_details -> why each skipped job failed (id, title, company,
                        error_type, error_message, attempts)
"""

from typing import Any, TypedDict

from models.job import Job, JobMatch


class PipelineState(TypedDict, total=False):
    profile: dict[str, Any]
    jobs: list[Job]
    filtered_jobs: list[Job]
    matched_jobs: list[JobMatch]
    ranked_jobs: list[JobMatch]
    skipped_jobs: int
    skipped_details: list[dict]
