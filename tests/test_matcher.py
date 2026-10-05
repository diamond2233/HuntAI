"""Tests for the OpenAI-powered matching stage (pipeline/matcher.py).

No real OpenAI calls are made -- the OpenAI client is mocked throughout.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APITimeoutError, OpenAIError, RateLimitError
from pydantic import ValidationError

from models.job import Job
from pipeline.matcher import (
    DEFAULT_MAX_WORKERS,
    MATCH_TEMPERATURE,
    MAX_DESCRIPTION_CHARS,
    MAX_RATE_LIMIT_RETRY_ATTEMPTS,
    MAX_RETRY_ATTEMPTS,
    RATE_LIMIT_WAIT_CAP_SECONDS,
    _filter_skills_against_profile,
    _MatchResult,
    _RateLimitState,
    _rate_limit_wait_seconds,
    match_jobs,
)


def _job(title="Backend Engineer", company="Acme", description="Some description.") -> Job:
    return Job(
        id="1",
        title=title,
        company=company,
        location="Bangalore, India",
        description=description,
        url="https://example.com/1",
        source="greenhouse",
    )


def _profile() -> dict:
    return {
        "roles": ["Backend Engineer", "Software Engineer"],
        "experience": {"min_years": 0, "max_years": 2},
        "skills": ["Python", "SQL", "Git"],
    }


def _fake_completion(parsed=None, refusal=None):
    message = SimpleNamespace(parsed=parsed, refusal=refusal)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _mock_client(parsed_result) -> MagicMock:
    client = MagicMock()
    client.chat.completions.parse.return_value = _fake_completion(parsed=parsed_result)
    return client


@pytest.fixture(autouse=True)
def openai_model_env(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")


def test_valid_job_produces_one_job_match():
    result = _MatchResult(score=82, matched_skills=["Python"], missing_skills=["Go"], rationale="Good fit.")
    client = _mock_client(result)

    with patch("pipeline.matcher.OpenAI", return_value=client):
        matches = match_jobs([_job()], _profile())

    assert len(matches) == 1


def test_score_is_numeric_and_in_range():
    result = _MatchResult(score=75.5, rationale="Solid.")
    client = _mock_client(result)

    with patch("pipeline.matcher.OpenAI", return_value=client):
        matches = match_jobs([_job()], _profile())

    assert isinstance(matches[0].score, (int, float))
    assert 0 <= matches[0].score <= 100


def test_matched_and_missing_skills_are_lists():
    result = _MatchResult(score=60, matched_skills=["Python", "SQL"], missing_skills=["Go"], rationale="Ok.")
    client = _mock_client(result)

    with patch("pipeline.matcher.OpenAI", return_value=client):
        matches = match_jobs([_job()], _profile())

    assert matches[0].matched_skills == ["Python", "SQL"]
    assert matches[0].missing_skills == ["Go"]


def test_rationale_is_present():
    result = _MatchResult(score=60, rationale="Because reasons.")
    client = _mock_client(result)

    with patch("pipeline.matcher.OpenAI", return_value=client):
        matches = match_jobs([_job()], _profile())

    assert matches[0].rationale == "Because reasons."


def test_original_job_is_preserved_in_job_match():
    job = _job()
    result = _MatchResult(score=60, rationale="Ok.")
    client = _mock_client(result)

    with patch("pipeline.matcher.OpenAI", return_value=client):
        matches = match_jobs([job], _profile())

    assert matches[0].job is job


def test_profile_information_is_sent_to_the_model():
    result = _MatchResult(score=60, rationale="Ok.")
    client = _mock_client(result)

    with patch("pipeline.matcher.OpenAI", return_value=client):
        match_jobs([_job()], _profile())

    user_message = client.chat.completions.parse.call_args.kwargs["messages"][1]["content"]
    assert "Backend Engineer" in user_message
    assert "Python" in user_message
    assert "0-2 years" in user_message


def test_job_title_and_description_are_sent_to_the_model():
    result = _MatchResult(score=60, rationale="Ok.")
    client = _mock_client(result)
    job = _job(title="Platform Engineer", description="Own our Kubernetes platform.")

    with patch("pipeline.matcher.OpenAI", return_value=client):
        match_jobs([job], _profile())

    user_message = client.chat.completions.parse.call_args.kwargs["messages"][1]["content"]
    assert "Platform Engineer" in user_message
    assert "Kubernetes platform" in user_message


def test_long_description_is_truncated():
    result = _MatchResult(score=60, rationale="Ok.")
    client = _mock_client(result)
    long_description = "x" * (MAX_DESCRIPTION_CHARS + 500)
    job = _job(description=long_description)

    with patch("pipeline.matcher.OpenAI", return_value=client):
        match_jobs([job], _profile())

    user_message = client.chat.completions.parse.call_args.kwargs["messages"][1]["content"]
    sent_description = user_message.split("Description: ", 1)[1]
    assert len(sent_description) == MAX_DESCRIPTION_CHARS


def test_empty_job_list_returns_empty_without_calling_openai():
    with patch("pipeline.matcher.OpenAI") as MockOpenAI:
        matches = match_jobs([], _profile())

    assert matches == []
    MockOpenAI.assert_not_called()


def test_non_retryable_openai_failure_is_skipped_not_raised():
    # OpenAIError("boom") is a generic, non-retryable error (not a rate
    # limit/timeout/5xx) -- it fails the job once and is not retried. With
    # failure isolation, that no longer aborts match_jobs(): the job is
    # skipped and counted instead.
    client = MagicMock()
    client.chat.completions.parse.side_effect = OpenAIError("boom")
    stats = {}

    with patch("pipeline.matcher.OpenAI", return_value=client):
        matches = match_jobs([_job(title="Backend Engineer", company="Acme")], _profile(), stats=stats)

    assert matches == []
    assert stats["skipped_jobs"] == 1


def test_invalid_structured_output_is_skipped_not_raised():
    try:
        _MatchResult(score=999, rationale="bad")
        validation_error = None
    except ValidationError as exc:
        validation_error = exc
    assert validation_error is not None

    client = MagicMock()
    client.chat.completions.parse.side_effect = validation_error
    stats = {}

    with patch("pipeline.matcher.OpenAI", return_value=client):
        matches = match_jobs([_job()], _profile(), stats=stats)

    assert matches == []
    assert stats["skipped_jobs"] == 1


def test_refusal_is_skipped_not_raised():
    client = MagicMock()
    client.chat.completions.parse.return_value = _fake_completion(refusal="cannot help with this")
    stats = {}

    with patch("pipeline.matcher.OpenAI", return_value=client):
        matches = match_jobs([_job()], _profile(), stats=stats)

    assert matches == []
    assert stats["skipped_jobs"] == 1


# --- skipped_details: every skip must say why -----------------------------

def test_skipped_details_records_id_title_company_for_non_retryable_error():
    client = MagicMock()
    client.chat.completions.parse.side_effect = OpenAIError("boom")
    job = _job(title="Backend Engineer", company="Acme")
    stats = {}

    with patch("pipeline.matcher.OpenAI", return_value=client):
        match_jobs([job], _profile(), stats=stats)

    assert len(stats["skipped_details"]) == 1
    detail = stats["skipped_details"][0]
    assert detail["id"] == job.id
    assert detail["title"] == "Backend Engineer"
    assert detail["company"] == "Acme"
    assert detail["error_type"] == "OpenAIError"
    assert "boom" in detail["error_message"]
    assert detail["attempts"] == 1


def test_skipped_details_records_attempts_and_error_type_after_retry_exhaustion():
    client = MagicMock()
    client.chat.completions.parse.side_effect = [_timeout_error()] * (MAX_RETRY_ATTEMPTS + 1)
    stats = {}

    with (
        patch("pipeline.matcher.time.sleep"),
        patch("pipeline.matcher.OpenAI", return_value=client),
    ):
        match_jobs([_job()], _profile(), stats=stats)

    detail = stats["skipped_details"][0]
    assert detail["error_type"] == "APITimeoutError"
    assert detail["attempts"] == MAX_RETRY_ATTEMPTS + 1


def test_skipped_details_records_refusal_as_its_own_error_type():
    client = MagicMock()
    client.chat.completions.parse.return_value = _fake_completion(refusal="cannot help with this")
    stats = {}

    with patch("pipeline.matcher.OpenAI", return_value=client):
        match_jobs([_job()], _profile(), stats=stats)

    detail = stats["skipped_details"][0]
    assert detail["error_type"] == "Refusal"
    assert detail["attempts"] == 1
    assert "cannot help with this" in detail["error_message"]


def test_skipped_details_records_missing_structured_result_as_its_own_error_type():
    client = MagicMock()
    client.chat.completions.parse.return_value = _fake_completion(parsed=None)
    stats = {}

    with patch("pipeline.matcher.OpenAI", return_value=client):
        match_jobs([_job()], _profile(), stats=stats)

    detail = stats["skipped_details"][0]
    assert detail["error_type"] == "NoStructuredResult"
    assert detail["attempts"] == 1


def test_multiple_skipped_jobs_each_get_their_own_detail_entry():
    def fake_parse(*, model, messages, response_format, **kwargs):
        raise OpenAIError("boom")

    client = MagicMock()
    client.chat.completions.parse.side_effect = fake_parse
    jobs = [_job(title="Job A"), _job(title="Job B")]
    stats = {}

    with patch("pipeline.matcher.OpenAI", return_value=client):
        match_jobs(jobs, _profile(), stats=stats)

    assert stats["skipped_jobs"] == 2
    assert {d["title"] for d in stats["skipped_details"]} == {"Job A", "Job B"}


def test_no_skips_results_in_empty_skipped_details():
    result = _MatchResult(score=60, rationale="Ok.")
    client = _mock_client(result)
    stats = {}

    with patch("pipeline.matcher.OpenAI", return_value=client):
        match_jobs([_job()], _profile(), stats=stats)

    assert stats["skipped_jobs"] == 0
    assert stats["skipped_details"] == []


def test_multiple_jobs_result_in_one_api_call_per_job():
    result = _MatchResult(score=60, rationale="Ok.")
    client = _mock_client(result)
    jobs = [_job(title="Job A"), _job(title="Job B"), _job(title="Job C")]

    with patch("pipeline.matcher.OpenAI", return_value=client):
        matches = match_jobs(jobs, _profile())

    assert len(matches) == 3
    assert client.chat.completions.parse.call_count == 3


def test_missing_model_env_var_raises_clear_error(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    with patch("pipeline.matcher.OpenAI") as MockOpenAI:
        with pytest.raises(RuntimeError):
            match_jobs([_job()], _profile())

    MockOpenAI.assert_not_called()


def _timeout_error() -> APITimeoutError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return APITimeoutError(request=request)


def _rate_limit_error(message: str = "Rate limit reached.", retry_after_header: str | None = None) -> RateLimitError:
    headers = {"Retry-After": retry_after_header} if retry_after_header is not None else {}
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code=429, headers=headers, request=request)
    return RateLimitError(message, response=response, body=None)


def test_retries_on_temporary_error_then_succeeds():
    result = _MatchResult(score=70, rationale="Ok after retry.")
    client = MagicMock()
    client.chat.completions.parse.side_effect = [
        _timeout_error(),
        _timeout_error(),
        _fake_completion(parsed=result),
    ]

    with (
        patch("pipeline.matcher.time.sleep") as mock_sleep,
        patch("pipeline.matcher.OpenAI", return_value=client),
    ):
        matches = match_jobs([_job()], _profile())

    assert len(matches) == 1
    assert matches[0].score == 70
    assert client.chat.completions.parse.call_count == 3
    # Slept before each of the 2 retries, not after the final success.
    assert mock_sleep.call_count == 2


def test_retry_backoff_is_exponential():
    client = MagicMock()
    client.chat.completions.parse.side_effect = [_timeout_error(), _timeout_error(), _timeout_error(), _timeout_error()]

    with (
        patch("pipeline.matcher.time.sleep") as mock_sleep,
        patch("pipeline.matcher.OpenAI", return_value=client),
    ):
        match_jobs([_job()], _profile())

    slept_for = [call.args[0] for call in mock_sleep.call_args_list]
    assert slept_for == [1.0, 2.0, 4.0]


# --- rate-limit handling: Retry-After, message parsing, cap, fallback -----

def test_wait_time_is_taken_from_retry_after_header(monkeypatch):
    monkeypatch.setattr("pipeline.matcher.random.uniform", lambda a, b: 0.0)
    exc = _rate_limit_error(message="no hint here", retry_after_header="2.5")

    wait_seconds = _rate_limit_wait_seconds(exc, retry_number=0)

    assert wait_seconds == 2.5


def test_wait_time_is_parsed_from_message_when_no_header():
    exc = _rate_limit_error(message="Please try again in 245ms.")

    wait_seconds = _rate_limit_wait_seconds(exc, retry_number=0)

    assert 0.245 <= wait_seconds <= 0.245 + 1.0  # + jitter (0-1s)


def test_wait_time_is_parsed_from_message_in_seconds():
    exc = _rate_limit_error(message="Please try again in 3.2s.")

    wait_seconds = _rate_limit_wait_seconds(exc, retry_number=0)

    assert 3.2 <= wait_seconds <= 3.2 + 1.0


def test_wait_time_falls_back_to_exponential_backoff_when_no_hint(monkeypatch):
    monkeypatch.setattr("pipeline.matcher.random.uniform", lambda a, b: 0.0)
    exc = _rate_limit_error(message="rate limited, no timing info")

    assert _rate_limit_wait_seconds(exc, retry_number=0) == 1.0
    assert _rate_limit_wait_seconds(exc, retry_number=1) == 2.0
    assert _rate_limit_wait_seconds(exc, retry_number=2) == 4.0


def test_wait_time_is_capped_at_60_seconds(monkeypatch):
    monkeypatch.setattr("pipeline.matcher.random.uniform", lambda a, b: 1.0)
    exc = _rate_limit_error(retry_after_header="1000")

    wait_seconds = _rate_limit_wait_seconds(exc, retry_number=0)

    assert wait_seconds == RATE_LIMIT_WAIT_CAP_SECONDS


def test_shared_cooldown_makes_a_shorter_wait_defer_to_a_longer_one(monkeypatch):
    fake_now = [100.0]
    monkeypatch.setattr("pipeline.matcher.time.monotonic", lambda: fake_now[0])
    sleeps = []
    monkeypatch.setattr("pipeline.matcher.time.sleep", lambda seconds: sleeps.append(seconds))

    state = _RateLimitState()

    # One worker hits a 429 with a long wait.
    state.extend(5.0)
    # A second worker hits a 429 too, but with a much shorter wait -- the
    # shared resume time must not shrink because of it.
    state.extend(0.5)

    state.wait()

    assert sleeps == [5.0]
    assert state.hits == 2


def test_rate_limit_hits_and_wait_seconds_are_recorded_in_stats(monkeypatch):
    monkeypatch.setattr("pipeline.matcher.random.uniform", lambda a, b: 0.0)
    result = _MatchResult(score=60, rationale="Ok.")
    client = MagicMock()
    client.chat.completions.parse.side_effect = [
        _rate_limit_error(retry_after_header="0.01"),
        _fake_completion(parsed=result),
    ]
    stats = {}

    with patch("pipeline.matcher.OpenAI", return_value=client):
        match_jobs([_job()], _profile(), stats=stats)

    assert stats["rate_limit_hits"] == 1
    assert stats["rate_limit_wait_seconds"] > 0


def test_rate_limit_allows_up_to_six_retries_then_skips(monkeypatch):
    monkeypatch.setattr("pipeline.matcher.random.uniform", lambda a, b: 0.0)
    client = MagicMock()
    client.chat.completions.parse.side_effect = [
        _rate_limit_error(retry_after_header="0.01")
    ] * (MAX_RATE_LIMIT_RETRY_ATTEMPTS + 1)
    stats = {}

    with patch("pipeline.matcher.OpenAI", return_value=client):
        matches = match_jobs([_job()], _profile(), stats=stats)

    assert matches == []
    assert stats["skipped_jobs"] == 1
    assert stats["skipped_details"][0]["attempts"] == MAX_RATE_LIMIT_RETRY_ATTEMPTS + 1
    assert stats["skipped_details"][0]["error_type"] == "RateLimitError"
    assert client.chat.completions.parse.call_count == MAX_RATE_LIMIT_RETRY_ATTEMPTS + 1


def test_job_still_failing_after_all_retries_is_skipped_and_counted():
    client = MagicMock()
    client.chat.completions.parse.side_effect = [_timeout_error()] * (3 + 1)
    stats = {}

    with (
        patch("pipeline.matcher.time.sleep"),
        patch("pipeline.matcher.OpenAI", return_value=client),
    ):
        matches = match_jobs([_job(title="Bad Job")], _profile(), stats=stats)

    assert matches == []
    assert stats["skipped_jobs"] == 1
    assert client.chat.completions.parse.call_count == 4


def test_one_failing_job_does_not_block_the_others():
    good_result = _MatchResult(score=80, rationale="Fine.")

    def fake_parse(*, model, messages, response_format, **kwargs):
        user_message = messages[1]["content"]
        if "Bad Job" in user_message:
            raise _timeout_error()
        return _fake_completion(parsed=good_result)

    client = MagicMock()
    client.chat.completions.parse.side_effect = fake_parse
    jobs = [_job(title="Bad Job"), _job(title="Good Job")]
    stats = {}

    with (
        patch("pipeline.matcher.time.sleep"),
        patch("pipeline.matcher.OpenAI", return_value=client),
    ):
        matches = match_jobs(jobs, _profile(), stats=stats)

    assert len(matches) == 1
    assert matches[0].job.title == "Good Job"
    assert stats["skipped_jobs"] == 1


def test_empty_job_list_sets_zero_skipped_in_stats():
    stats = {}
    with patch("pipeline.matcher.OpenAI") as MockOpenAI:
        matches = match_jobs([], _profile(), stats=stats)

    assert matches == []
    assert stats["skipped_jobs"] == 0
    MockOpenAI.assert_not_called()


def test_results_keep_input_order_regardless_of_which_call_finishes_first():
    def fake_parse(*, model, messages, response_format, **kwargs):
        user_message = messages[1]["content"]
        if "Slow Job" in user_message:
            time.sleep(0.05)
            score = 10
        else:
            score = 90
        return _fake_completion(parsed=_MatchResult(score=score, rationale="ok"))

    client = MagicMock()
    client.chat.completions.parse.side_effect = fake_parse
    jobs = [_job(title="Slow Job"), _job(title="Fast Job")]

    with patch("pipeline.matcher.OpenAI", return_value=client):
        matches = match_jobs(jobs, _profile(), max_workers=2)

    assert [match.job.title for match in matches] == ["Slow Job", "Fast Job"]


def test_client_is_created_with_sdk_retries_disabled():
    # The SDK retries 429/5xx/timeouts itself by default -- our own retry
    # loop is the only retry layer we want, so the client must be created
    # with max_retries=0 to turn the SDK's off.
    result = _MatchResult(score=60, rationale="Ok.")
    client = _mock_client(result)

    with patch("pipeline.matcher.OpenAI", return_value=client) as MockOpenAI:
        match_jobs([_job()], _profile())

    MockOpenAI.assert_called_once_with(max_retries=0)


def test_openai_call_uses_temperature_zero():
    result = _MatchResult(score=60, rationale="Ok.")
    client = _mock_client(result)

    with patch("pipeline.matcher.OpenAI", return_value=client):
        match_jobs([_job()], _profile())

    assert client.chat.completions.parse.call_args.kwargs["temperature"] == MATCH_TEMPERATURE
    assert MATCH_TEMPERATURE == 0


# --- trustworthy skill lists: code-level enforcement, not just a prompt ---

def test_invented_matched_skill_is_dropped():
    matched, missing, dropped_matched, dropped_missing = _filter_skills_against_profile(
        matched_skills=["Python", "Ruby"],
        missing_skills=[],
        profile_skills=["Python", "SQL"],
    )

    assert matched == ["Python"]
    assert dropped_matched == 1
    assert dropped_missing == 0


def test_profile_skill_listed_as_missing_is_dropped():
    matched, missing, dropped_matched, dropped_missing = _filter_skills_against_profile(
        matched_skills=[],
        missing_skills=["Java", "Docker"],
        profile_skills=["Java", "Python"],
    )

    assert missing == ["Docker"]
    assert dropped_missing == 1
    assert dropped_matched == 0


def test_valid_matched_and_missing_skills_are_kept():
    matched, missing, dropped_matched, dropped_missing = _filter_skills_against_profile(
        matched_skills=["Python", "SQL"],
        missing_skills=["Docker"],
        profile_skills=["Python", "SQL", "Git"],
    )

    assert matched == ["Python", "SQL"]
    assert missing == ["Docker"]
    assert dropped_matched == 0
    assert dropped_missing == 0


def test_case_and_whitespace_differences_are_handled():
    matched, missing, dropped_matched, dropped_missing = _filter_skills_against_profile(
        matched_skills=["  python  ", "SQL"],
        missing_skills=["  PYTHON "],
        profile_skills=["Python", "SQL"],
    )

    # Both matched entries count as real profile skills despite case/
    # whitespace differences, so neither is dropped...
    assert matched == ["  python  ", "SQL"]
    assert dropped_matched == 0
    # ...and "PYTHON" (with different case/whitespace) is still recognized
    # as the profile skill "Python", so it's dropped from missing_skills.
    assert missing == []
    assert dropped_missing == 1


def test_exact_comparison_only_no_fuzzy_matching():
    # "Golang" is not literally "Go" -- no fuzzy/equivalence matching here,
    # even though the two are conceptually related (that's the prompt's
    # job to recognize when deciding relevance, not this function's).
    matched, missing, dropped_matched, dropped_missing = _filter_skills_against_profile(
        matched_skills=["Golang"],
        missing_skills=[],
        profile_skills=["Go"],
    )

    assert matched == []
    assert dropped_matched == 1


def test_dropped_skill_counts_are_reported_in_stats():
    result = _MatchResult(
        score=70,
        matched_skills=["Python", "Ruby"],
        missing_skills=["Java", "Docker"],
        rationale="ok",
    )
    client = _mock_client(result)
    profile = {"roles": [], "skills": ["Python", "Java"]}
    stats = {}

    with patch("pipeline.matcher.OpenAI", return_value=client):
        matches = match_jobs([_job()], profile, stats=stats)

    assert matches[0].matched_skills == ["Python"]
    assert matches[0].missing_skills == ["Docker"]
    assert stats["dropped_matched_skills"] == 1
    assert stats["dropped_missing_skills"] == 1


def test_dropped_skill_counts_default_to_zero_for_empty_job_list():
    stats = {}
    with patch("pipeline.matcher.OpenAI"):
        match_jobs([], _profile(), stats=stats)

    assert stats["dropped_matched_skills"] == 0
    assert stats["dropped_missing_skills"] == 0


def test_dropped_skill_counts_accumulate_across_multiple_jobs():
    def fake_parse(*, model, messages, response_format, **kwargs):
        return _fake_completion(
            parsed=_MatchResult(
                score=50, matched_skills=["Python", "Ruby"], missing_skills=[], rationale="ok"
            )
        )

    client = MagicMock()
    client.chat.completions.parse.side_effect = fake_parse
    profile = {"roles": [], "skills": ["Python"]}
    jobs = [_job(title="Job A"), _job(title="Job B")]
    stats = {}

    with patch("pipeline.matcher.OpenAI", return_value=client):
        match_jobs(jobs, profile, stats=stats)

    assert stats["dropped_matched_skills"] == 2


def test_default_max_workers_is_four():
    assert DEFAULT_MAX_WORKERS == 4


def test_max_workers_is_configurable():
    result = _MatchResult(score=60, rationale="Ok.")
    client = _mock_client(result)

    with (
        patch("pipeline.matcher.OpenAI", return_value=client),
        patch("pipeline.matcher.ThreadPoolExecutor", wraps=ThreadPoolExecutor) as mock_executor,
    ):
        match_jobs([_job()], _profile(), max_workers=3)

    mock_executor.assert_called_once_with(max_workers=3)


def test_max_workers_defaults_to_four_when_not_specified():
    result = _MatchResult(score=60, rationale="Ok.")
    client = _mock_client(result)

    with (
        patch("pipeline.matcher.OpenAI", return_value=client),
        patch("pipeline.matcher.ThreadPoolExecutor", wraps=ThreadPoolExecutor) as mock_executor,
    ):
        match_jobs([_job()], _profile())

    mock_executor.assert_called_once_with(max_workers=4)
