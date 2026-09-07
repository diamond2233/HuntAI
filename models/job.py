"""Pydantic data models for HuntAI.

Job is the normalized representation that every collector (Greenhouse, Lever, ...)
must convert its raw source data into. The rest of the pipeline only ever works
with Job objects and never needs to know which source produced them.

JobMatch is the structured result produced later by the OpenAI matching stage.
"""

from datetime import date
from typing import Optional

from pydantic import BaseModel


class Job(BaseModel):
    """A normalized job posting, regardless of which source it came from."""

    id: str
    title: str
    company: str
    location: Optional[str] = None
    description: Optional[str] = None
    url: str
    source: str
    posted_date: Optional[date] = None


class JobMatch(BaseModel):
    """The structured output of matching a Job against the user's profile."""

    job: Job
    score: float
    matched_skills: list[str] = []
    missing_skills: list[str] = []
    rationale: str
