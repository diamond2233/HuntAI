"""Tests for the OpenAI-powered matching stage (pipeline/matcher.py).

No real OpenAI calls are made -- the OpenAI client is mocked throughout.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAIError
from pydantic import ValidationError

from models.job import Job
from pipeline.matcher import MAX_DESCRIPTION_CHARS, _MatchResult, match_jobs


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


def test_openai_failure_raises_clear_runtime_error():
    client = MagicMock()
    client.chat.completions.parse.side_effect = OpenAIError("boom")

    with patch("pipeline.matcher.OpenAI", return_value=client):
        with pytest.raises(RuntimeError) as exc_info:
            match_jobs([_job(title="Backend Engineer", company="Acme")], _profile())

    assert "Backend Engineer" in str(exc_info.value)
    assert "Acme" in str(exc_info.value)


def test_invalid_structured_output_raises_clear_runtime_error():
    try:
        _MatchResult(score=999, rationale="bad")
        validation_error = None
    except ValidationError as exc:
        validation_error = exc
    assert validation_error is not None

    client = MagicMock()
    client.chat.completions.parse.side_effect = validation_error

    with patch("pipeline.matcher.OpenAI", return_value=client):
        with pytest.raises(RuntimeError):
            match_jobs([_job()], _profile())


def test_refusal_raises_clear_runtime_error():
    client = MagicMock()
    client.chat.completions.parse.return_value = _fake_completion(refusal="cannot help with this")

    with patch("pipeline.matcher.OpenAI", return_value=client):
        with pytest.raises(RuntimeError):
            match_jobs([_job()], _profile())


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
