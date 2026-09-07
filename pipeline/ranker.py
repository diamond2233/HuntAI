"""Ranking stage (deterministic, no LLM).

Sorts already-scored JobMatch objects by score, highest first. All of the
judgment (skill relevance, experience fit, semantic suitability) already
happened in the matching stage and is captured in JobMatch.score -- ranking
is just ordering numbers, which doesn't need an LLM.
"""

from models.job import JobMatch


def rank_jobs(matched_jobs: list[JobMatch]) -> list[JobMatch]:
    """Return a new list of matches sorted by score, highest first.

    Python's sorted() is a stable sort, so matches with equal scores keep
    their original relative order. The input list itself is never modified.
    """
    return sorted(matched_jobs, key=lambda match: match.score, reverse=True)
