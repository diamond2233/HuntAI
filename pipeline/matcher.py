"""OpenAI-powered semantic job matching stage.

After deterministic filtering (excluded company, title, location) and
URL-based deduplication, this is the one place in the pipeline that talks to
an LLM. For each remaining Job, exactly one OpenAI call evaluates how well it
fits the user's profile -- skill relevance, experience compatibility, and
overall fit -- and returns a structured JobMatch. Pydantic validates the
model's output before it can enter the pipeline state.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError

from models.job import Job, JobMatch

load_dotenv()

# Job descriptions can be very long; this keeps a single request from
# blowing up in size. No summarization -- just a hard character cutoff.
MAX_DESCRIPTION_CHARS = 12000

SYSTEM_PROMPT = """You are a job-profile matching evaluator for a job search assistant.

You will be given a candidate PROFILE (target roles, years of experience,
skills) and a single JOB (title, company, location, description). Decide how
well this job matches the candidate's profile.

Rules:
- Base your judgment only on the PROFILE and JOB information provided. Do
  not invent skills, facts, or requirements that aren't stated or clearly
  implied.
- Distinguish skills the job clearly requires from technologies it merely
  mentions in passing.
- Treat clearly equivalent skills as the same thing (e.g. "Generative AI"
  and "GenAI"; "Go" and "Golang").
- matched_skills: profile skills that are genuinely relevant to this job.
- missing_skills: only meaningful gaps -- profile skills the job clearly
  needs but that aren't reflected in the job. Do not list every profile
  skill the description happens not to mention.
- Evaluate experience compatibility against the candidate's experience
  range. If the job does not clearly state an experience requirement, say so
  in the rationale instead of guessing a number.
- The job title alone doesn't decide relevance -- consider whether the
  description itself is a strong fit for the candidate's target roles, even
  if the title differs somewhat.
- Give a concise rationale (a few sentences) explaining the score.
- Return only the requested structured fields.

Score from 0 to 100. The score should weigh required/relevant skills,
experience fit, role relevance, and overall suitability together -- NOT
simply the count of matching skills (a job needing 3 skills the candidate
has all of can outscore a job listing 10 skills the candidate mostly lacks).
  90-100 = exceptional match
  75-89  = strong match
  60-74  = reasonable match
  40-59  = weak match
  0-39   = poor match
"""


class _MatchResult(BaseModel):
    """Structured output requested from the LLM.

    JobMatch is not used directly as the response_format because it embeds
    the full Job object -- the model should only judge the job, never
    generate or echo job data itself. The real JobMatch is assembled by
    combining this result with the original Job we already have.
    """

    score: float = Field(ge=0, le=100)
    matched_skills: list[str] = []
    missing_skills: list[str] = []
    rationale: str


def _format_experience(experience: dict | None) -> str:
    if not experience:
        return "Not specified"
    min_years = experience.get("min_years")
    max_years = experience.get("max_years")
    if min_years is None and max_years is None:
        return "Not specified"
    return f"{min_years if min_years is not None else 0}-{max_years if max_years is not None else '?'} years"


def _build_user_message(job: Job, profile: dict) -> str:
    """Build the input message: only what affects semantic matching.

    The job's id/url/source don't affect matching, so they're left out.
    """
    description = job.description or "Not provided"
    if len(description) > MAX_DESCRIPTION_CHARS:
        description = description[:MAX_DESCRIPTION_CHARS]

    return (
        "PROFILE:\n"
        f"Target roles: {', '.join(profile.get('roles') or [])}\n"
        f"Experience: {_format_experience(profile.get('experience'))}\n"
        f"Skills: {', '.join(profile.get('skills') or [])}\n"
        "\n"
        "JOB:\n"
        f"Title: {job.title}\n"
        f"Company: {job.company}\n"
        f"Location: {job.location or 'Not specified'}\n"
        f"Description: {description}"
    )


def _match_job(client: OpenAI, model: str, job: Job, profile: dict) -> JobMatch:
    user_message = _build_user_message(job, profile)

    try:
        completion = client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format=_MatchResult,
        )
    except (OpenAIError, ValidationError) as exc:
        raise RuntimeError(
            f"OpenAI matching call failed for job '{job.title}' at '{job.company}': {exc}"
        ) from exc

    message = completion.choices[0].message

    if message.refusal:
        raise RuntimeError(
            f"OpenAI refused to evaluate job '{job.title}' at '{job.company}': {message.refusal}"
        )

    result = message.parsed
    if result is None:
        raise RuntimeError(
            f"OpenAI returned no structured result for job '{job.title}' at '{job.company}'."
        )

    return JobMatch(
        job=job,
        score=result.score,
        matched_skills=result.matched_skills,
        missing_skills=result.missing_skills,
        rationale=result.rationale,
    )


def match_jobs(jobs: list[Job], profile: dict) -> list[JobMatch]:
    """Evaluate each job against the profile with one OpenAI call per job."""
    if not jobs:
        return []

    model = os.getenv("OPENAI_MODEL")
    if not model:
        raise RuntimeError("OPENAI_MODEL is not set. Add it to .env before running the matcher.")

    client = OpenAI()
    return [_match_job(client, model, job, profile) for job in jobs]
