"""LangGraph pipeline skeleton.

Defines the node sequence for HuntAI:

    START -> collect -> filter -> deduplicate -> match -> rank -> export -> END

Every node now has real logic: collect_node reads source configuration from
state["profile"] and runs the configured collectors; filter_node applies the
deterministic filter rules; deduplicate_node removes duplicate postings by
URL; match_node calls the OpenAI matcher; rank_node sorts matches by score;
export_node writes the ranked results to Markdown/CSV/JSON files.
"""

from pathlib import Path

from langgraph.graph import END, START, StateGraph

from collectors.apify import ApifyCollector
from collectors.greenhouse import GreenhouseCollector
from collectors.lever import LeverCollector
from exporters.csv import export_csv
from exporters.json import export_json
from exporters.markdown import export_markdown
from models.job import Job
from pipeline.dedup import deduplicate
from pipeline.filters import apply_filters
from pipeline.matcher import match_jobs
from pipeline.ranker import rank_jobs
from pipeline.state import PipelineState

OUTPUT_DIR = Path("output")


def collect_node(state: PipelineState) -> PipelineState:
    """Run every enabled, configured collector and gather their jobs.

    This node owns *orchestration* (which collectors to run, for which
    boards/companies) -- it never talks to Greenhouse or Lever directly. All
    HTTP/parsing logic stays inside GreenhouseCollector and LeverCollector.

    If a configured source fails (e.g. a bad board token, a network error),
    the collector raises and that exception propagates out of this node and
    out of graph.invoke() -- collection failures are not swallowed.
    """
    profile = state.get("profile") or {}
    sources = profile.get("sources") or {}
    greenhouse_config = sources.get("greenhouse") or {}
    lever_config = sources.get("lever") or {}
    apify_config = sources.get("apify") or {}

    jobs: list[Job] = []

    if greenhouse_config.get("enabled"):
        for board in greenhouse_config.get("boards") or []:
            token = board.get("token")
            if not token:
                continue
            collector = GreenhouseCollector(board_token=token, company=board.get("company"))
            jobs.extend(collector.collect())

    if lever_config.get("enabled"):
        for company in lever_config.get("companies") or []:
            slug = company.get("slug")
            if not slug:
                continue
            collector = LeverCollector(company_slug=slug, company=company.get("company"))
            jobs.extend(collector.collect())

    if apify_config.get("enabled"):
        collector = ApifyCollector(
            roles=profile.get("roles") or [],
            locations=profile.get("locations") or [],
            max_items=apify_config.get("max_items", 100),
            max_pages=apify_config.get("max_pages", 2),
        )
        jobs.extend(collector.collect())

    return {"jobs": jobs}


def filter_node(state: PipelineState) -> PipelineState:
    """Apply the deterministic filter (excluded company, role, location)."""
    profile = state.get("profile") or {}
    jobs = state.get("jobs") or []
    filtered_jobs = apply_filters(jobs, profile)
    return {"filtered_jobs": filtered_jobs}


def deduplicate_node(state: PipelineState) -> PipelineState:
    """Remove duplicate postings (by URL) from filtered_jobs.

    Writes back into filtered_jobs rather than a new state field: dedup is
    just a refinement of the same "jobs that survived filtering" list that
    match_node consumes next, so a separate field would add a key without
    adding meaning.
    """
    filtered_jobs = state.get("filtered_jobs") or []
    deduplicated_jobs = deduplicate(filtered_jobs)
    return {"filtered_jobs": deduplicated_jobs}


def match_node(state: PipelineState) -> PipelineState:
    """Run each deduplicated job through the OpenAI matcher."""
    profile = state.get("profile") or {}
    filtered_jobs = state.get("filtered_jobs") or []
    matched_jobs = match_jobs(filtered_jobs, profile)
    return {"matched_jobs": matched_jobs}


def rank_node(state: PipelineState) -> PipelineState:
    """Sort matched jobs by score, highest first."""
    matched_jobs = state.get("matched_jobs") or []
    ranked_jobs = rank_jobs(matched_jobs)
    return {"ranked_jobs": ranked_jobs}


def export_node(state: PipelineState) -> PipelineState:
    """Write ranked_jobs to Markdown, CSV, and JSON files.

    This node only orchestrates *which* exporters run and *where* they
    write -- all formatting logic lives in exporters/*.py, keeping LangGraph
    orchestration separate from output formatting.
    """
    ranked_jobs = state.get("ranked_jobs") or []

    export_markdown(ranked_jobs, OUTPUT_DIR / "jobs.md")
    export_csv(ranked_jobs, OUTPUT_DIR / "jobs.csv")
    export_json(ranked_jobs, OUTPUT_DIR / "jobs.json")

    return {"ranked_jobs": ranked_jobs}


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("collect", collect_node)
    graph.add_node("filter", filter_node)
    graph.add_node("deduplicate", deduplicate_node)
    graph.add_node("match", match_node)
    graph.add_node("rank", rank_node)
    graph.add_node("export", export_node)

    graph.add_edge(START, "collect")
    graph.add_edge("collect", "filter")
    graph.add_edge("filter", "deduplicate")
    graph.add_edge("deduplicate", "match")
    graph.add_edge("match", "rank")
    graph.add_edge("rank", "export")
    graph.add_edge("export", END)

    return graph.compile()
