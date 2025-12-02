import os

import arxiv
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from cache import paper_id_from_url, load_cache, save_cache
from llm_client import LLMClient
import json
from datetime import datetime, timedelta, timezone

def get_24h_window():
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=24)
    return window_start


class State(dict):
    """L'état partagé entre les noeuds du graphe."""
    query: str
    raw_papers: list
    analyzed: list
    scored: list
    
    
def build_arxiv_query() -> str:
    categories = ["cs.CL", "cs.LG", "cs.AI", "cs.IR", "cs.MA"]

    cat_query = " OR ".join([f"cat:{c}" for c in categories])

    full_query = f"({cat_query})"
    return full_query




def search_arxiv(state: State):
    print("🔎 Recherche ArXIV...")
    window_start = get_24h_window()
    query = build_arxiv_query()
    results = arxiv.Search(
        query=query,
        max_results=200,
        sort_by=arxiv.SortCriterion.SubmittedDate
    ).results()

    papers = []
    for r in results:
        if r.published < window_start:
            continue
        papers.append({
            "title": r.title,
            "abstract": r.summary,
            "url": r.entry_id,
            "published": str(r.published),
            "authors": [a.name for a in r.authors]
        })

    state["raw_papers"] = papers
    return state




llm = LLMClient()

def analyze_papers(state: State):
    print("🧠 Analyse des papiers...")

    analyzed = []

    for p in state["raw_papers"]:
        pid = paper_id_from_url(p["url"])
        cached = load_cache(pid)

        if cached and "analysis" in cached:
            print(f"⚡ Cache hit: {pid} (analysis)")
            p["analysis"] = cached["analysis"]
            analyzed.append(p)
            continue

        print(f"🔍 Analyse LLM: {pid}")

        prompt = f"""
        Analyse le papier suivant :
        Titre : {p['title']}
        Résumé : {p['abstract']}
        
        Extrait :
        - Les contributions principales
        - Les innovations techniques
        - Les applications possibles
        - La difficulté de reproduction
        - Un résumé en 5 lignes
        """

        analysis = llm.invoke(prompt).content
        p["analysis"] = analysis

        # Save to cache
        cached = cached or {}
        cached.update(p)
        save_cache(pid, cached)

        analyzed.append(p)

    state["analyzed"] = analyzed
    return state




def score_papers(state: State):
    print("📊 Scoring des papiers...")

    scored = []

    for p in state["analyzed"]:
        pid = paper_id_from_url(p["url"])
        cached = load_cache(pid)

        # ---- Cache hit : score déjà présent ----
        if cached and "score" in cached:
            print(f"⚡ Cache hit: {pid} (score)")
            p["score"] = cached["score"]
            scored.append(p)
            continue

        print(f"🏷️ Scoring LLM: {pid}")

        prompt = f"""
        À partir de l'analyse suivante :
        {p["analysis"]}

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

        score_json = llm.invoke(prompt).content
        p["score"] = score_json

        cached = cached or {}
        cached.update({"score": score_json})
        save_cache(pid, cached)

        scored.append(p)

    state["scored"] = scored
    return state




workflow = StateGraph(State)

workflow.add_node("search_arxiv", search_arxiv)
workflow.add_node("analyze_papers", analyze_papers)
workflow.add_node("score_papers", score_papers)

workflow.set_entry_point("search_arxiv")

workflow.add_edge("search_arxiv", "analyze_papers")
workflow.add_edge("analyze_papers", "score_papers")
workflow.add_edge("score_papers", END)

graph = workflow.compile()



if __name__ == "__main__":
    user_query = ""
    
    result = graph.invoke({"query": user_query})
    
    scored_papers = []
    for p in result["scored"]:
        try:
            score_data = json.loads(p["score"])
            p["score_json"] = score_data  
            p["score_value"] = float(score_data.get("score_global", 0))
        except Exception:
            p["score_json"] = {}
            p["score_value"] = 0.0
        scored_papers.append(p)
        

    scored_papers = sorted(scored_papers, key=lambda p: p["score_value"])

    print("\n🎉 RESULTAT FINAL — Trié par score (ascendant)\n")

    for p in scored_papers:
        print("-----")
        print("Title :", p["title"])
        print("Summary :", p["abstract"])
        print("Score global :", p["score_value"])
        print("Score complet JSON :", p["score_json"])
        print("URL :", p["url"])
        print()
