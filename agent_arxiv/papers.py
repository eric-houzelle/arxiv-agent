import json
from typing import Any, Dict, List

from .state import State


def parse_score(score_payload: str) -> Dict[str, Any]:
    try:
        return json.loads(score_payload)
    except json.JSONDecodeError:
        return {}


def collect_scored_papers(state: State) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for paper in state.get("scored", []):
        score_data = parse_score(paper.get("score", "{}"))
        score_value = float(score_data.get("score_global", 0)) if score_data else 0.0
        paper["score_json"] = score_data
        paper["score_value"] = score_value
        scored.append(paper)
    return sorted(scored, key=lambda paper: paper["score_value"], reverse=True)


def format_linkedin_brief(papers: List[Dict[str, Any]]) -> str:
    sections: List[str] = []
    for idx, paper in enumerate(papers, start=1):
        authors = ", ".join(paper.get("authors") or []) or "Unknown"
        scores = paper.get("score_json") or {}
        score_parts = []
        for label, key in (
            ("Orig", "originalite"),
            ("Tech", "impact"),
            ("Repro", "repro"),
            ("Short", "potentiel"),
            ("Global", "score_global"),
        ):
            value = scores.get(key)
            if value is not None:
                score_parts.append(f"{label}:{value}")

        summary_source = paper.get("analysis") or paper.get("abstract") or ""
        summary = summary_source.strip()
        if len(summary) > 1200:
            summary = summary[:1200] + "..."

        block_lines = [
            f"Paper #{idx}: {paper.get('title', 'Unknown title')}",
            f"Authors: {authors}",
        ]
        if paper.get("url"):
            block_lines.append(f"URL: {paper['url']}")
        if score_parts:
            block_lines.append(f"Scores: {', '.join(score_parts)}")
        if summary:
            block_lines.append(f"Analysis: {summary}")
        sections.append("\n".join(block_lines))

    return "\n\n".join(sections)
