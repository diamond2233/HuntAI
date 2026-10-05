"""Tests for the deterministic filtering stage (pipeline/filters.py).

Pure function tests -- no network, no LLM, just Job objects and profile
dicts.
"""

from models.job import Job
from pipeline.filters import (
    DEFAULT_EXCLUDE_TITLE_WORDS,
    apply_filters,
    has_excluded_title_word,
    is_excluded_company,
    matches_location,
    matches_role,
)


def _job(title="Software Engineer", company="Acme", location="Bangalore, India"):
    return Job(
        id=title,
        title=title,
        company=company,
        location=location,
        url="https://example.com/1",
        source="greenhouse",
    )


# --- excluded companies ------------------------------------------------------

def test_excluded_company_is_removed():
    assert is_excluded_company("TCS", ["TCS", "Infosys"]) is True


def test_excluded_company_matching_is_case_insensitive():
    assert is_excluded_company("tcs", ["TCS"]) is True
    assert is_excluded_company("Tcs", ["TCS"]) is True


def test_non_excluded_company_is_not_excluded():
    assert is_excluded_company("Acme", ["TCS", "Infosys"]) is False


def test_no_excluded_companies_configured_excludes_nothing():
    assert is_excluded_company("TCS", []) is False


# --- role/title matching ------------------------------------------------------

def test_matching_role_is_kept():
    assert matches_role("Backend Engineer", ["Backend Engineer"]) is True


def test_non_matching_role_is_removed():
    assert matches_role("Product Manager", ["Backend Engineer", "Software Engineer"]) is False


def test_role_matching_works_for_longer_title():
    assert matches_role("Senior Backend Engineer", ["Backend Engineer"]) is True
    assert matches_role("Software Engineer II", ["Software Engineer"]) is True


def test_role_matching_is_case_insensitive():
    assert matches_role("senior backend engineer", ["Backend Engineer"]) is True


def test_management_title_is_not_incorrectly_matched():
    assert matches_role("Software Engineering Manager", ["Software Engineer"]) is False


def test_no_roles_configured_does_not_reject_by_title():
    assert matches_role("Anything At All", []) is True


# --- role/title recall net (generic technical titles) -------------------------

PROFILE_ROLES = ["Software Engineer", "Backend Engineer", "SDE", "AI Engineer", "GenAI Engineer"]


def test_technical_title_variants_are_allowed():
    variants = [
        "Software Development Engineer",
        "Machine Learning Engineer",
        "ML Engineer",
        "Applied AI Engineer",
        "Generative AI Engineer",
        "Data Engineer",
        "Backend Developer",
        "Software Developer",
        "Platform Engineer",
        "AI/ML Engineer",
    ]
    for title in variants:
        assert matches_role(title, PROFILE_ROLES) is True, f"expected {title!r} to match"


def test_obviously_unrelated_titles_are_still_rejected():
    unrelated = ["Sales Executive", "HR Business Partner", "Content Writer", "Office Administrator"]
    for title in unrelated:
        assert matches_role(title, PROFILE_ROLES) is False, f"expected {title!r} to be rejected"


def test_management_titles_are_not_accidentally_accepted_by_recall_net():
    management_titles = [
        "Engineering Manager",
        "Director of Engineering",
        "VP of Engineering",
        "Head of Engineering",
        "Chief Technology Officer",
        "Engineer Manager",  # unusual phrasing containing "Engineer" as a whole word
    ]
    for title in management_titles:
        assert matches_role(title, PROFILE_ROLES) is False, f"expected {title!r} to be rejected"


def test_configured_roles_still_match_exactly_after_recall_net_added():
    assert matches_role("Software Engineer", PROFILE_ROLES) is True
    assert matches_role("Senior Backend Engineer", PROFILE_ROLES) is True
    assert matches_role("SDE", PROFILE_ROLES) is True
    assert matches_role("AI Engineer", PROFILE_ROLES) is True


# --- location matching --------------------------------------------------------

def test_location_matching_is_case_insensitive():
    assert matches_location("bangalore, india", ["Bangalore"]) is True


def test_location_matching_substring_examples():
    assert matches_location("Bangalore, India", ["Bangalore"]) is True
    assert matches_location("Hyderabad", ["Hyderabad"]) is True
    assert matches_location("Remote - India", ["Remote"]) is True
    assert matches_location("New Delhi / Noida", ["Noida"]) is True
    assert matches_location("Mumbai", ["Bangalore", "Hyderabad"]) is False


def test_multiple_configured_locations_work():
    locations = ["Bangalore", "Hyderabad", "Pune", "Noida", "Gurgaon", "Remote"]
    assert matches_location("Gurgaon, India", locations) is True
    assert matches_location("Chennai", locations) is False


def test_missing_location_is_rejected_when_locations_configured():
    assert matches_location(None, ["Bangalore"]) is False


def test_empty_location_configuration_does_not_reject_jobs():
    assert matches_location(None, []) is True
    assert matches_location("Anywhere", []) is True


# --- location aliases (Bengaluru -> Bangalore) ---------------------------------
# Evidenced from real rejection data: 124 jobs were rejected with location
# "Bengaluru, India" even though the profile configures "Bangalore".

def test_bengaluru_matches_configured_bangalore():
    assert matches_location("Bengaluru, India", ["Bangalore"]) is True
    assert matches_location("Bengaluru", ["Bangalore"]) is True


def test_bengaluru_does_not_match_a_different_configured_city():
    assert matches_location("Bengaluru, India", ["Hyderabad"]) is False


def test_bangalore_spelling_still_matches_after_alias_added():
    assert matches_location("Bangalore, India", ["Bangalore"]) is True


# --- "Remote" must mean India, not just any remote job ------------------------
# Evidenced from real rejection data: once global companies were added, 215
# of 250 filtered jobs were "Remote - California"/"Remote, USA"/etc -- not
# India -- passing only because "remote" is a substring of those strings.

def test_bare_remote_matches_configured_remote():
    assert matches_location("Remote", ["Remote"]) is True


def test_remote_india_matches_configured_remote():
    assert matches_location("Remote - India", ["Remote"]) is True
    assert matches_location("Remote, India", ["Remote"]) is True


def test_remote_outside_india_does_not_match_configured_remote():
    assert matches_location("Remote - California", ["Remote"]) is False
    assert matches_location("Remote, USA", ["Remote"]) is False
    assert matches_location("Remote - Canada", ["Remote"]) is False


def test_remote_outside_india_is_rejected_even_with_other_locations_configured():
    locations = ["Bangalore", "Hyderabad", "Pune", "Noida", "Gurgaon", "Remote"]
    assert matches_location("Remote - California", locations) is False
    assert matches_location("Toronto, Ontario, Canada", locations) is False


def test_non_remote_city_configured_is_unaffected_by_remote_rule():
    # A job that isn't remote at all and doesn't match any configured city
    # must still be rejected -- the Remote-specific rule shouldn't leak into
    # ordinary city matching.
    assert matches_location("San Francisco, California", ["Bangalore", "Remote"]) is False


# --- seniority filter (exclude_title_words) ------------------------------------

def test_each_default_excluded_word_is_rejected():
    for word in DEFAULT_EXCLUDE_TITLE_WORDS:
        title = f"Backend Engineer - {word}"
        assert has_excluded_title_word(title, DEFAULT_EXCLUDE_TITLE_WORDS) is True, f"expected {word!r} to be rejected"


def test_excluded_word_in_the_middle_of_a_title_is_rejected():
    assert has_excluded_title_word("Backend Engineer, Staff Level", DEFAULT_EXCLUDE_TITLE_WORDS) is True


def test_sr_with_a_period_is_rejected():
    assert has_excluded_title_word("Sr. Backend Engineer", DEFAULT_EXCLUDE_TITLE_WORDS) is True


def test_staffing_is_not_rejected_by_staff():
    # Word-boundary matching: "staff" must not match inside "Staffing".
    assert has_excluded_title_word("Staffing Coordinator", DEFAULT_EXCLUDE_TITLE_WORDS) is False


def test_intermediate_backend_engineer_is_kept_by_apply_filters():
    profile = {"roles": [], "locations": [], "excluded_companies": []}
    job = _job(title="Intermediate Backend Engineer")

    result = apply_filters([job], profile)

    assert len(result) == 1


def test_software_engineer_is_kept_by_apply_filters():
    profile = {"roles": [], "locations": [], "excluded_companies": []}
    job = _job(title="Software Engineer")

    result = apply_filters([job], profile)

    assert len(result) == 1


def test_default_rejects_senior_titles_via_apply_filters():
    profile = {"roles": [], "locations": [], "excluded_companies": []}
    jobs = [_job(title="Senior Backend Engineer"), _job(title="Software Engineer")]

    result = apply_filters(jobs, profile)

    assert [job.title for job in result] == ["Software Engineer"]


def test_empty_exclude_title_words_disables_the_rule():
    profile = {
        "roles": [],
        "locations": [],
        "excluded_companies": [],
        "exclude_title_words": [],
    }
    job = _job(title="Senior Backend Engineer")

    result = apply_filters([job], profile)

    assert len(result) == 1


def test_custom_exclude_title_words_overrides_the_default():
    profile = {
        "roles": [],
        "locations": [],
        "excluded_companies": [],
        "exclude_title_words": ["intern"],
    }
    jobs = [_job(title="Senior Backend Engineer"), _job(title="Backend Engineer Intern")]

    result = apply_filters(jobs, profile)

    # "Senior" is no longer excluded (the custom list doesn't include it);
    # "Intern" now is, even though it's not in the default list at all.
    assert [job.title for job in result] == ["Senior Backend Engineer"]


# --- apply_filters end-to-end --------------------------------------------------

def test_apply_filters_removes_excluded_company():
    profile = {"excluded_companies": ["TCS"], "roles": [], "locations": []}
    jobs = [_job(company="TCS"), _job(company="Acme")]

    result = apply_filters(jobs, profile)

    assert [job.company for job in result] == ["Acme"]


def test_apply_filters_ignores_experience():
    # No experience handling in apply_filters -- a narrow configured
    # experience range must not remove an otherwise-matching job.
    profile = {
        "excluded_companies": [],
        "roles": ["Software Engineer"],
        "locations": [],
        "experience": {"min_years": 5, "max_years": 10},
    }
    jobs = [_job(title="Software Engineer")]

    result = apply_filters(jobs, profile)

    assert len(result) == 1


def test_apply_filters_ignores_skills():
    # No skills-based filtering -- zero skill overlap still passes.
    profile = {
        "excluded_companies": [],
        "roles": ["Software Engineer"],
        "locations": [],
        "skills": ["Python", "Go"],
    }
    job = _job(title="Software Engineer")
    job.description = "Requires COBOL and Fortran only."

    result = apply_filters([job], profile)

    assert len(result) == 1


def test_apply_filters_handles_empty_profile_safely():
    jobs = [_job()]
    result = apply_filters(jobs, {})
    assert result == jobs


def test_apply_filters_handles_missing_description_safely():
    # apply_filters never reads job.description -- a job with none (the
    # common real-world case for some sources) must not crash or be rejected
    # for that reason alone.
    profile = {"excluded_companies": [], "roles": ["Software Engineer"], "locations": []}
    job = _job(title="Software Engineer")
    assert job.description is None

    result = apply_filters([job], profile)

    assert len(result) == 1
    assert result[0].description is None
