import json
from datetime import datetime, timedelta, timezone
from math import e
from typing import Any, Dict, List
from io import BytesIO
import arxiv
from langgraph.graph import END, StateGraph

from cache import load_cache, paper_id_from_url, save_cache
from llm_client import LLMClient
import requests
from pypdf import PdfReader

def get_24h_window():
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=24)
    return window_start


class State(dict):
    """État partagé entre les nœuds du graphe."""
    query: str
    raw_papers: list
    analyzed: list
    scored: list


DEFAULT_CATEGORIES = ["cs.CL", "cs.LG", "cs.AI", "cs.IR", "cs.MA"]


def build_arxiv_query(categories: List[str] | None = None) -> str:
    selected = categories or DEFAULT_CATEGORIES
    cat_query = " OR ".join([f"cat:{cat}" for cat in selected])
    return f"({cat_query})"




def search_arxiv(state: State):
    print("🔎 Recherche ArXIV...")
    window_start = get_24h_window()
    query = state.get("query") or build_arxiv_query()
    results = arxiv.Search(
        query=query,
        max_results=200,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    ).results()

    papers: List[Dict[str, Any]] = []
    for result in results:
        if result.published < window_start:
            continue
        papers.append(
            {
                "title": result.title,
                "abstract": result.summary,
                "url": result.entry_id,
                "pdf_url": result.pdf_url,
                "published": str(result.published),
                "authors": [author.name for author in result.authors],
            }
        )

    state["raw_papers"] = papers
    return state


def _download_pdf(pdf_url: str) -> bytes:
    response = requests.get(pdf_url, timeout=30)
    response.raise_for_status()
    return response.content


def _extract_pdf_content(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages_text: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text:
            pages_text.append(text)
    return "\n".join(pages_text)


def fetch_pdf_content(state: State):
    print("📄 Extraction du contenu PDF...")

    papers_with_content: List[Dict[str, Any]] = []

    for paper in state.get("raw_papers", []):
        paper_id = paper_id_from_url(paper["url"])
        cached = load_cache(paper_id)

        if cached and "content" in cached:
            print(f"⚡ Cache hit: {paper_id} (content)")
            paper["content"] = cached["content"]
            papers_with_content.append(paper)
            continue

        pdf_url = paper.get("pdf_url")
        if not pdf_url:
            print(f"⚠️ Aucun PDF trouvé pour {paper_id}")
            papers_with_content.append(paper)
            continue

        try:
            pdf_bytes = _download_pdf(pdf_url)
            content = _extract_pdf_content(pdf_bytes)
            _attach_cached_field(paper, paper_id, cached, "content", content)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Impossible de récupérer le PDF {paper_id}: {exc}")

        papers_with_content.append(paper)

    state["raw_papers"] = papers_with_content
    return state

llm = LLMClient()


def _build_analysis_prompt(paper: Dict[str, Any]) -> str:
    content = paper.get("content")
    content_excerpt = ""

    if content:
        truncated = content[:4000]
        content_excerpt = f"\nContenu (extrait) :\n{truncated}\n"

    return f"""
    Analyse le papier suivant :
    Titre : {paper['title']}
    Résumé : {paper['abstract']}
    
    {content_excerpt}
    Extrait :
    - Les contributions principales
    - Les innovations techniques
    - Les applications possibles
    - La difficulté de reproduction
    - Un résumé en 5 lignes
    """


def _build_score_prompt(analysis: str) -> str:
    return f"""
    À partir de l'analyse suivante :
    {analysis}

    Donne un score entre 0 et 10 sur :
    - originalité
    - impact technique
    - facilité de reproduction
    - potentiel à court terme

    Puis un score global final entre 0 et 10.
    Retourne au format JSON :
    {{
      "originalite": x,
      "impact": x,
      "repro": x,
      "potentiel": x,
      "score_global": x
    }}
    """


def _attach_cached_field(
    paper: Dict[str, Any],
    paper_id: str,
    cached: Dict[str, Any] | None,
    field: str,
    value: Any,
):
    paper[field] = value
    cached = cached or {}
    cached[field] = value
    save_cache(paper_id, cached)


def analyze_papers(state: State):
    print("🧠 Analyse des papiers...")

    analyzed: List[Dict[str, Any]] = []

    for paper in state.get("raw_papers", []):
        paper_id = paper_id_from_url(paper["url"])
        cached = load_cache(paper_id)

        if cached and "analysis" in cached:
            print(f"⚡ Cache hit: {paper_id} (analysis)")
            paper["analysis"] = cached["analysis"]
            analyzed.append(paper)
            continue

        print(f"🔍 Analyse LLM: {paper_id}")
        prompt = _build_analysis_prompt(paper)
        analysis = llm.invoke(prompt).content
        _attach_cached_field(paper, paper_id, cached, "analysis", analysis)
        analyzed.append(paper)

    state["analyzed"] = analyzed
    return state




def score_papers(state: State):
    print("📊 Scoring des papiers...")

    scored: List[Dict[str, Any]] = []

    for paper in state.get("analyzed", []):
        paper_id = paper_id_from_url(paper["url"])
        cached = load_cache(paper_id)

        if cached and "score" in cached:
            print(f"⚡ Cache hit: {paper_id} (score)")
            paper["score"] = cached["score"]
            scored.append(paper)
            continue

        print(f"🏷️ Scoring LLM: {paper_id}")
        prompt = _build_score_prompt(paper["analysis"])
        score_json = llm.invoke(prompt).content
        _attach_cached_field(paper, paper_id, cached, "score", score_json)
        scored.append(paper)

    state["scored"] = scored
    return state




workflow = StateGraph(State)

workflow.add_node("search_arxiv", search_arxiv)
workflow.add_node("fetch_pdf_content", fetch_pdf_content)
workflow.add_node("analyze_papers", analyze_papers)
workflow.add_node("score_papers", score_papers)

workflow.set_entry_point("search_arxiv")

workflow.add_edge("search_arxiv", "fetch_pdf_content")
workflow.add_edge("fetch_pdf_content", "analyze_papers")
workflow.add_edge("analyze_papers", "score_papers")
workflow.add_edge("score_papers", END)

graph = workflow.compile()


def _parse_score(score_payload: str) -> Dict[str, Any]:
    try:
        return json.loads(score_payload)
    except json.JSONDecodeError:
        return {}


def _collect_scored_papers(state: State) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for paper in state.get("scored", []):
        score_data = _parse_score(paper.get("score", "{}"))
        score_value = float(score_data.get("score_global", 0)) if score_data else 0.0
        paper["score_json"] = score_data
        paper["score_value"] = score_value
        scored.append(paper)
    return sorted(scored, key=lambda paper: paper["score_value"], reverse=True)


def main():
    result = graph.invoke({"query": ""})
    scored_papers = _collect_scored_papers(result)
    for paper in scored_papers:
        scores = _parse_score(paper.get("score", "{}"))
        if scores != {}:
            print(f"{paper['score_value']:.1f} - {paper['title']} (Imp:{scores['impact']}/Rep:{scores['repro']}/Pot:{scores['potentiel']}/Ori:{scores['originalite']})")
            print(f"     {paper['url']}")
            print(f"     {paper['abstract']}")
            print(f" ")

if __name__ == "__main__":
    main()
