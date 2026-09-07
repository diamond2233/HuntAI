"""HuntAI entry point.

Runs the collection stage of the pipeline end to end: loads config/profile.yaml,
builds the LangGraph graph, and invokes it. 
"""

import yaml

from pipeline.graph import build_graph


def load_profile(path: str = "config/profile.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    profile = load_profile()

    graph = build_graph()
    result = graph.invoke({"profile": profile, "jobs": []})

    jobs = result.get("jobs", [])
    print(f"Collected {len(jobs)} jobs.")


if __name__ == "__main__":
    main()
