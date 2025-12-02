from langgraph.graph import END, StateGraph

from .nodes import (
    analyze_papers,
    fetch_pdf_content,
    score_papers,
    search_arxiv,
    write_linkedin_post,
)
from .state import State


def build_workflow() -> StateGraph:
    workflow = StateGraph(State)

    workflow.add_node("search_arxiv", search_arxiv)
    workflow.add_node("fetch_pdf_content", fetch_pdf_content)
    workflow.add_node("analyze_papers", analyze_papers)
    workflow.add_node("score_papers", score_papers)
    workflow.add_node("write_linkedin_post", write_linkedin_post)

    workflow.set_entry_point("search_arxiv")

    workflow.add_edge("search_arxiv", "fetch_pdf_content")
    workflow.add_edge("fetch_pdf_content", "analyze_papers")
    workflow.add_edge("analyze_papers", "score_papers")
    workflow.add_edge("score_papers", "write_linkedin_post")
    workflow.add_edge("write_linkedin_post", END)
    return workflow


graph = build_workflow().compile()


def run_workflow(query: str = ""):
    return graph.invoke({"query": query})
