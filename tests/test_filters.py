"""Tests for the deterministic filtering stage (pipeline/filters.py).

Pure function tests -- no network, no LLM, just Job objects and profile
dicts.
"""

from models.job import Job
from pipeline.filters import apply_filters, is_excluded_company, matches_location, matches_role


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
