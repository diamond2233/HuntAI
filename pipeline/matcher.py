"""OpenAI-powered semantic job matching stage.

After deterministic filtering (excluded company, title, location) and
URL-based deduplication, this is the one place in the pipeline that talks to
an LLM. For each remaining Job, exactly one OpenAI call evaluates how well it
fits the user's profile -- skill relevance, experience compatibility, and
overall fit -- and returns a structured JobMatch. Pydantic validates the
model's output before it can enter the pipeline state.

Jobs are matched concurrently with a thread pool (OpenAI calls are I/O-bound,
so threads -- not processes -- are enough to overlap them). Temporary errors
are retried rather than failing the whole run:

- Rate limits (429) wait for however long the server actually asked for
  (shared across all workers -- see _RateLimitState), up to 6 retries.
- Timeouts, connection errors, and 5xx keep the original, simpler
  exponential backoff, up to 3 retries.

A job that still fails after retries is skipped rather than failing the
whole run.
"""

import logging
import os
import random
import re
import threading
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
# overlap many calls without needing multiprocessing. 4 is chosen over a
# higher number because real data (results/h1e_ratelimit.json) showed 8
# workers hit the account's rate limit 51 times and still lost a job, while
# 4 workers finished the same 165 jobs with zero rate-limit hits and only
# ~15% slower -- past this point, OpenAI's per-minute token cap is the
# bottleneck, not thread count, so more workers stops paying off.
DEFAULT_MAX_WORKERS = 4

# Timeouts, connection errors, and 5xx: unchanged from before -- a fixed
# exponential backoff, since there's no server-provided wait time to use
# for these.
OTHER_RETRYABLE_ERRORS = (APITimeoutError, InternalServerError, APIConnectionError)
MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 1.0

# Rate limits (429) get their own, longer retry budget, since a per-minute
# token limit can easily outlast the backoff above.
MAX_RATE_LIMIT_RETRY_ATTEMPTS = 6
RATE_LIMIT_WAIT_CAP_SECONDS = 60.0
RATE_LIMIT_JITTER_SECONDS = 1.0
RATE_LIMIT_FALLBACK_BACKOFF_SECONDS = 1.0

# Matches the "try again in X s" / "try again in X ms" text OpenAI's own
# rate-limit error message includes, e.g. "Please try again in 245ms."
RATE_LIMIT_MESSAGE_PATTERN = re.compile(r"try again in\s+([\d.]+)\s*(ms|s)\b", re.IGNORECASE)

# 0 asks the model for its most deterministic sampling. This meaningfully
# reduces (but does not eliminate) run-to-run drift in scores and skill
# lists -- OpenAI documents that reproducibility is "not guaranteed" even
# at temperature=0 or with a fixed seed, due to backend non-determinism
# outside this code's control. See docs/H1_EXPLAINED.md's H1i section.
MATCH_TEMPERATURE = 0

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


class MatchFailure(RuntimeError):
    """Raised when a job could not be matched, carrying enough detail for
    the caller to report exactly why it's being skipped -- not just that it
    was.
    """

    def __init__(self, message: str, *, error_type: str, attempts: int):
        super().__init__(message)
        self.error_type = error_type
        self.attempts = attempts


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


class _RateLimitState:
    """Shared across every worker in one match_jobs() call.

    When any worker hits a 429, it's a sign the whole account just went
    over its per-minute token budget -- not that this one job's request was
    special. So instead of each worker independently retrying (and likely
    hitting the same still-active limit again), they all pause until one
    shared "resume_at" timestamp has passed. Protected by a plain lock;
    nothing fancier than that.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._resume_at = 0.0
        self.hits = 0
        self.total_wait_seconds = 0.0

    def extend(self, wait_seconds: float) -> None:
        """Record a 429 and push the shared resume time forward if this
        wait would end later than whatever's currently set."""
        with self._lock:
            self.hits += 1
            candidate = time.monotonic() + wait_seconds
            if candidate > self._resume_at:
                self._resume_at = candidate

    def wait(self) -> None:
        """Sleep until the shared resume time has passed, and record how
        long this call actually waited."""
        with self._lock:
            resume_at = self._resume_at
        remaining = resume_at - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
            with self._lock:
                self.total_wait_seconds += remaining


def _rate_limit_wait_seconds_from_header(exc: RateLimitError) -> float | None:
    """The server's own Retry-After header, if it sent one."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    value = headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rate_limit_wait_seconds_from_message(exc: RateLimitError) -> float | None:
    """OpenAI's rate-limit errors describe the wait in the message text
    itself (e.g. "Please try again in 245ms."), even without a header."""
    match = RATE_LIMIT_MESSAGE_PATTERN.search(str(exc))
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).lower()
    return amount / 1000 if unit == "ms" else amount


def _rate_limit_wait_seconds(exc: RateLimitError, retry_number: int) -> float:
    """How long to wait before retrying a rate-limited call.

    Prefers the server's own guidance over guessing: the Retry-After
    header, then the "try again in Xs/Xms" text in the error message, and
    only falls back to exponential backoff if neither is present. A little
    random jitter is added so several workers that all hit the limit at
    once don't all retry at the exact same instant, and the total is
    capped so a bad parse/huge header value can't block things for ages.
    """
    wait_seconds = _rate_limit_wait_seconds_from_header(exc)
    if wait_seconds is None:
        wait_seconds = _rate_limit_wait_seconds_from_message(exc)
    if wait_seconds is None:
        wait_seconds = RATE_LIMIT_FALLBACK_BACKOFF_SECONDS * (2**retry_number)

    jitter = random.uniform(0, RATE_LIMIT_JITTER_SECONDS)
    return min(wait_seconds + jitter, RATE_LIMIT_WAIT_CAP_SECONDS)


def _call_with_retries(
    client: OpenAI, model: str, job: Job, profile: dict, rate_limit_state: _RateLimitState
):
    """Call the OpenAI API, retrying temporary errors.

    Rate limits (429) and other temporary errors (timeout/connection/5xx)
    are retried with different strategies -- see the module docstring.
    Any other error is raised immediately, since retrying it wouldn't help.
    """
    user_message = _build_user_message(job, profile)

    rate_limit_retries = 0
    other_retries = 0

    while True:
        try:
            return client.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format=_MatchResult,
                temperature=MATCH_TEMPERATURE,
            )
        except RateLimitError as exc:
            attempts = 1 + rate_limit_retries + other_retries
            if rate_limit_retries >= MAX_RATE_LIMIT_RETRY_ATTEMPTS:
                raise MatchFailure(
                    f"OpenAI matching call failed for job '{job.title}' at "
                    f"'{job.company}' after {attempts} attempts (rate limited): {exc}",
                    error_type=type(exc).__name__,
                    attempts=attempts,
                ) from exc
            wait_seconds = _rate_limit_wait_seconds(exc, rate_limit_retries)
            rate_limit_state.extend(wait_seconds)
            logger.warning(
                "Rate limited for job '%s' at '%s' (retry %d/%d) -- waiting (shared cooldown, up to %.1fs).",
                job.title,
                job.company,
                rate_limit_retries + 1,
                MAX_RATE_LIMIT_RETRY_ATTEMPTS,
                wait_seconds,
            )
            rate_limit_state.wait()
            rate_limit_retries += 1
        except OTHER_RETRYABLE_ERRORS as exc:
            attempts = 1 + rate_limit_retries + other_retries
            if other_retries >= MAX_RETRY_ATTEMPTS:
                raise MatchFailure(
                    f"OpenAI matching call failed for job '{job.title}' at "
                    f"'{job.company}' after {attempts} attempts: {exc}",
                    error_type=type(exc).__name__,
                    attempts=attempts,
                ) from exc
            backoff_seconds = RETRY_BACKOFF_SECONDS * (2**other_retries)
            logger.warning(
                "OpenAI call failed for job '%s' at '%s' (attempt %d/%d, %s) -- retrying in %.1fs.",
                job.title,
                job.company,
                attempts,
                MAX_RETRY_ATTEMPTS + 1,
                type(exc).__name__,
                backoff_seconds,
            )
            time.sleep(backoff_seconds)
            other_retries += 1
        except (OpenAIError, ValidationError) as exc:
            attempts = 1 + rate_limit_retries + other_retries
            raise MatchFailure(
                f"OpenAI matching call failed for job '{job.title}' at '{job.company}': {exc}",
                error_type=type(exc).__name__,
                attempts=attempts,
            ) from exc


def _match_job(
    client: OpenAI, model: str, job: Job, profile: dict, rate_limit_state: _RateLimitState
) -> JobMatch:
    completion = _call_with_retries(client, model, job, profile, rate_limit_state)
    message = completion.choices[0].message

    if message.refusal:
        raise MatchFailure(
            f"OpenAI refused to evaluate job '{job.title}' at '{job.company}': {message.refusal}",
            error_type="Refusal",
            attempts=1,
        )

    result = message.parsed
    if result is None:
        raise MatchFailure(
            f"OpenAI returned no structured result for job '{job.title}' at '{job.company}'.",
            error_type="NoStructuredResult",
            attempts=1,
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
    were skipped (stats["skipped_jobs"]) and exactly why (stats
    ["skipped_details"], one entry per skipped job: id, title, company,
    error_type, error_message, attempts), plus how much rate limiting
    actually cost this run (stats["rate_limit_hits"], stats
    ["rate_limit_wait_seconds"]).
    """
    if not jobs:
        if stats is not None:
            stats["skipped_jobs"] = 0
            stats["skipped_details"] = []
            stats["rate_limit_hits"] = 0
            stats["rate_limit_wait_seconds"] = 0.0
        return []

    model = os.getenv("OPENAI_MODEL")
    if not model:
        raise RuntimeError("OPENAI_MODEL is not set. Add it to .env before running the matcher.")

    # The SDK itself retries 429/5xx/timeouts up to 2 times by default. Our
    # own retry loop in _call_with_retries() already handles that, so we
    # disable the SDK's retries here -- otherwise one failing call could be
    # attempted up to (our attempts) x (the SDK's 3 attempts) more times
    # than intended.
    client = OpenAI(max_retries=0)
    results: list[JobMatch | None] = [None] * len(jobs)
    skipped_details: list[dict] = []
    rate_limit_state = _RateLimitState()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_match_job, client, model, job, profile, rate_limit_state): index
            for index, job in enumerate(jobs)
        }
        for future, index in future_to_index.items():
            job = jobs[index]
            try:
                results[index] = future.result()
            except MatchFailure as exc:
                skipped_details.append(
                    {
                        "id": job.id,
                        "title": job.title,
                        "company": job.company,
                        "error_type": exc.error_type,
                        "error_message": str(exc),
                        "attempts": exc.attempts,
                    }
                )
                logger.error(
                    "Skipping job '%s' at '%s' -- failed after retries: %s",
                    job.title,
                    job.company,
                    exc,
                )

    if stats is not None:
        stats["skipped_jobs"] = len(skipped_details)
        stats["skipped_details"] = skipped_details
        stats["rate_limit_hits"] = rate_limit_state.hits
        stats["rate_limit_wait_seconds"] = round(rate_limit_state.total_wait_seconds, 3)

    return [match for match in results if match is not None]
