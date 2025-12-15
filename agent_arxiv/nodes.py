from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any, Dict, List

import arxiv
import requests
from pypdf import PdfReader

from cache import load_cache, paper_id_from_url, save_cache
from llm_client import LLMClient

from .config import (
    DEFAULT_CATEGORIES,
    LINKEDIN_CHARACTER_LIMIT,
    linkedin_language,
    linkedin_temperature,
)
from .logger import get_logger
from .papers import collect_scored_papers
from .prompts import (
    LINKEDIN_SYSTEM_PROMPT,
    build_analysis_prompt,
    build_linkedin_user_prompt,
    build_score_prompt,
)
from .state import State


llm = LLMClient()
logger = get_logger(__name__)


def get_24h_window():
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=72)
    return window_start


def build_arxiv_query(categories: List[str] | None = None) -> str:
    selected = categories or DEFAULT_CATEGORIES
    cat_query = " OR ".join([f"cat:{cat}" for cat in selected])
    return f"({cat_query})"


def search_arxiv(state: State):
    logger.info("Searching ArXiv...")
    window_start = get_24h_window()
    query = state.get("query") or build_arxiv_query()
    search_query = arxiv.Search(
        query=query,
        max_results=1000,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    results = list(search_query.results())
    logger.info("Results length: %s", len(results))

    papers: List[Dict[str, Any]] = []
    for result in results:
        if result.primary_category not in DEFAULT_CATEGORIES:
            continue
        if result.published < window_start:
            continue
        papers.append(
            {
                "title": result.title,
                "category": result.primary_category,
                "abstract": result.summary,
                "url": result.entry_id,
                "pdf_url": result.pdf_url,
                "published": str(result.published),
                "authors": [author.name for author in result.authors],
            }
        )

    logger.info("Papers length: %s", len(papers))
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


def fetch_pdf_content(state: State):
    logger.info("Fetching PDF contents...")

    papers_with_content: List[Dict[str, Any]] = []
    total = len(state.get("raw_papers", []))
    cache_hits = 0
    downloaded = 0
    missing_pdf = 0
    failures = 0

    for paper in state.get("raw_papers", []):
        paper_id = paper_id_from_url(paper["url"])
        cached = load_cache(paper_id)

        if cached and "content" in cached:
            logger.info("Cache hit: %s (content)", paper_id)
            paper["content"] = cached["content"]
            papers_with_content.append(paper)
            cache_hits += 1
            continue

        pdf_url = paper.get("pdf_url")
        if not pdf_url:
            logger.warning("No PDF found for %s", paper_id)
            papers_with_content.append(paper)
            missing_pdf += 1
            continue

        try:
            pdf_bytes = _download_pdf(pdf_url)
            content = _extract_pdf_content(pdf_bytes)
            _attach_cached_field(paper, paper_id, cached, "content", content)
            downloaded += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unable to fetch PDF %s", paper_id)
            failures += 1

        papers_with_content.append(paper)

    logger.info(
        "PDF stats - total: %s, cache hits: %s, downloaded: %s, "
        "missing pdf: %s, failures: %s",
        total,
        cache_hits,
        downloaded,
        missing_pdf,
        failures,
    )
    state["raw_papers"] = papers_with_content
    return state


def analyze_papers(state: State):
    logger.info("Analyzing papers...")

    analyzed: List[Dict[str, Any]] = []
    total = len(state.get("raw_papers", []))
    cache_hits = 0
    generated = 0

    for paper in state.get("raw_papers", []):
        paper_id = paper_id_from_url(paper["url"])
        cached = load_cache(paper_id)

        if cached and "analysis" in cached:
            logger.info("Cache hit: %s (analysis)", paper_id)
            paper["analysis"] = cached["analysis"]
            analyzed.append(paper)
            cache_hits += 1
            continue

        logger.info("🔍 LLM analysis: %s", paper_id)
        prompt = build_analysis_prompt(paper)
        analysis = llm.invoke(prompt).content
        _attach_cached_field(paper, paper_id, cached, "analysis", analysis)
        analyzed.append(paper)
        generated += 1

    logger.info(
        "Analysis stats - total: %s, cache hits: %s, generated: %s",
        total,
        cache_hits,
        generated,
    )
    state["analyzed"] = analyzed
    return state


def score_papers(state: State):
    logger.info("Scoring papers...")

    scored: List[Dict[str, Any]] = []
    total = len(state.get("analyzed", []))
    cache_hits = 0
    generated = 0

    for paper in state.get("analyzed", []):
        paper_id = paper_id_from_url(paper["url"])
        cached = load_cache(paper_id)

        if cached and "score" in cached:
            logger.info("⚡ Cache hit: %s (score)", paper_id)
            paper["score"] = cached["score"]
            scored.append(paper)
            cache_hits += 1
            continue

        logger.info("🏷️ LLM scoring: %s", paper_id)
        prompt = build_score_prompt(paper["analysis"])
        score_json = llm.invoke(prompt).content
        _attach_cached_field(paper, paper_id, cached, "score", score_json)
        scored.append(paper)
        generated += 1

    logger.info(
        "Score stats - total: %s, cache hits: %s, generated: %s",
        total,
        cache_hits,
        generated,
    )
    state["scored"] = scored
    return state


def write_linkedin_post(state: State):
    logger.info("Drafting LinkedIn post...")
    top_papers = collect_scored_papers(state)[:5]
    if not top_papers:
        logger.warning("No scored papers available for the LinkedIn post.")
        state["linkedin_post"] = ""
        return state

    language = linkedin_language()
    temperature = linkedin_temperature()
    user_prompt = build_linkedin_user_prompt(top_papers, language)
    messages = [
        {"role": "system", "content": LINKEDIN_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    post = llm.invoke_chat(messages, temperature=temperature).content
    state["linkedin_post"] = post
    state["top_papers"] = top_papers
    return state

