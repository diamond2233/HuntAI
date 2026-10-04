"""OpenAI-powered semantic job matching stage.

After deterministic filtering (excluded company, title, location) and
URL-based deduplication, this is the one place in the pipeline that talks to
an LLM. For each remaining Job, exactly one OpenAI call evaluates how well it
fits the user's profile -- skill relevance, experience compatibility, and
overall fit -- and returns a structured JobMatch. Pydantic validates the
model's output before it can enter the pipeline state.

Jobs are matched concurrently with a thread pool (OpenAI calls are I/O-bound,
so threads -- not processes -- are enough to overlap them). Temporary errors
(rate limits, timeouts, connection errors, 5xx) are retried with exponential
backoff; a job that still fails after retries is skipped rather than failing
the whole run.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
from pydantic import BaseModel, Field, ValidationError

from models.job import Job, JobMatch

load_dotenv()

logger = logging.getLogger(__name__)

# Job descriptions can be very long; this keeps a single request from
# blowing up in size. No summarization -- just a hard character cutoff.
MAX_DESCRIPTION_CHARS = 12000

# How many jobs to match at once. OpenAI calls spend almost all their time
# waiting on the network, so a modest number of threads is enough to
# overlap many calls without needing multiprocessing.
DEFAULT_MAX_WORKERS = 8

# Errors worth retrying -- ones likely to go away on their own if we wait and
# try again. Anything else (bad request, auth failure, refusal, invalid
# structured output, ...) is not retried, since trying again won't help.
RETRYABLE_ERRORS = (RateLimitError, APITimeoutError, InternalServerError, APIConnectionError)
MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 1.0

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


def _call_with_retries(client: OpenAI, model: str, job: Job, profile: dict):
    """Call the OpenAI API, retrying temporary errors with exponential backoff.

    Tries once, then retries up to MAX_RETRY_ATTEMPTS more times (so up to
    MAX_RETRY_ATTEMPTS + 1 attempts total) -- but only for RETRYABLE_ERRORS.
    Any other error is raised immediately, since retrying it wouldn't help.
    """
    user_message = _build_user_message(job, profile)

    for retry_number in range(MAX_RETRY_ATTEMPTS + 1):
        try:
            return client.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format=_MatchResult,
            )
        except RETRYABLE_ERRORS as exc:
            if retry_number == MAX_RETRY_ATTEMPTS:
                raise RuntimeError(
                    f"OpenAI matching call failed for job '{job.title}' at "
                    f"'{job.company}' after {retry_number + 1} attempts: {exc}"
                ) from exc
            backoff_seconds = RETRY_BACKOFF_SECONDS * (2**retry_number)
            logger.warning(
                "OpenAI call failed for job '%s' at '%s' (attempt %d/%d, %s) -- retrying in %.1fs.",
                job.title,
                job.company,
                retry_number + 1,
                MAX_RETRY_ATTEMPTS + 1,
                type(exc).__name__,
                backoff_seconds,
            )
            time.sleep(backoff_seconds)
        except (OpenAIError, ValidationError) as exc:
            raise RuntimeError(
                f"OpenAI matching call failed for job '{job.title}' at '{job.company}': {exc}"
            ) from exc


def _match_job(client: OpenAI, model: str, job: Job, profile: dict) -> JobMatch:
    completion = _call_with_retries(client, model, job, profile)
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


def match_jobs(
    jobs: list[Job],
    profile: dict,
    max_workers: int = DEFAULT_MAX_WORKERS,
    stats: dict | None = None,
) -> list[JobMatch]:
    """Evaluate each job against the profile, one OpenAI call per job.

    Jobs are matched concurrently (up to max_workers at once), but the
    returned list is always in the same order as the input `jobs`, no
    matter which call finishes first.

    A job that still fails after retries is logged and skipped instead of
    aborting the whole run. Pass a `stats` dict to find out how many jobs
    were skipped: stats["skipped_jobs"].
    """
    if not jobs:
        if stats is not None:
            stats["skipped_jobs"] = 0
        return []

    model = os.getenv("OPENAI_MODEL")
    if not model:
        raise RuntimeError("OPENAI_MODEL is not set. Add it to .env before running the matcher.")

    # The SDK itself retries 429/5xx/timeouts up to 2 times by default. Our
    # own retry loop in _call_with_retries() already handles that, so we
    # disable the SDK's retries here -- otherwise one failing call could be
    # attempted up to (our 4 attempts) x (the SDK's 3 attempts) = 12 times.
    client = OpenAI(max_retries=0)
    results: list[JobMatch | None] = [None] * len(jobs)
    skipped_jobs = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_match_job, client, model, job, profile): index
            for index, job in enumerate(jobs)
        }
        for future, index in future_to_index.items():
            job = jobs[index]
            try:
                results[index] = future.result()
            except RuntimeError as exc:
                skipped_jobs += 1
                logger.error(
                    "Skipping job '%s' at '%s' -- failed after retries: %s",
                    job.title,
                    job.company,
                    exc,
                )

    if stats is not None:
        stats["skipped_jobs"] = skipped_jobs

    return [match for match in results if match is not None]
