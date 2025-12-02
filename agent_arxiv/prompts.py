from typing import Any, Dict, List

from .config import (
    CRITERIA_PROMPT_FILES,
    PROMPTS_DIR,
    REPO_URL,
)
from .papers import format_linkedin_brief


def _load_text_file(path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def _load_criteria_prompts() -> List[tuple[str, str, str]]:
    prompts: List[tuple[str, str, str]] = []
    for key, label, filename in CRITERIA_PROMPT_FILES:
        path = PROMPTS_DIR / filename
        text = _load_text_file(path)
        prompts.append((key, label, text))
    return prompts


CRITERIA_PROMPTS = _load_criteria_prompts()


def format_criteria_guidelines() -> str:
    sections: List[str] = []
    for key, label, text in CRITERIA_PROMPTS:
        if not text:
            continue
        sections.append(f"{label} ({key})\n{text}")
    return "\n\n".join(sections)


def build_analysis_prompt(paper: Dict[str, Any]) -> str:
    content = paper.get("content")
    content_excerpt = ""

    if content:
        truncated = content[:100000]
        content_excerpt = f"\nContent excerpt:\n{truncated}\n"

    return f"""
    Analyze the following paper and produce the requested insights:
    Title: {paper['title']}
    Abstract: {paper['abstract']}
    START Content:
    {content_excerpt}
    
    END Content
    Return:
    - The main contributions
    - The technical innovations
    - Potential applications
    - How hard it is to reproduce
    - A five-line summary
    """


def build_score_prompt(analysis: str) -> str:
    criteria_guidelines = format_criteria_guidelines()
    return f"""
    Given the following analysis:
    {analysis}

    Use these guidelines for each criterion:
    {criteria_guidelines}

    Provide a score between 0 and 10 for:
    - originality
    - technical impact
    - reproducibility
    - short-term potential

    Also output a final global score between 0 and 10.
    Return JSON with this structure:
    {{
      "originalite": x,
      "impact": x,
      "repro": x,
      "potentiel": x,
      "score_global": x
    }}
    """


def _load_linkedin_system_prompt() -> str:
    default_prompt = (
        "You are a LinkedIn thought leader who helps AI enthusiasts"
        " understand cutting-edge research with enthusiasm and clarity."
    )
    path = PROMPTS_DIR / "linkedin_system.md"
    text = _load_text_file(path)
    return text or default_prompt


LINKEDIN_SYSTEM_PROMPT = _load_linkedin_system_prompt()


def build_linkedin_user_prompt(papers: List[Dict[str, Any]], language: str) -> str:
    brief = format_linkedin_brief(papers)
    paper_count = len(papers)
    return (
        f"Write a LinkedIn post in {language} that curates the top {paper_count} AI papers "
        "from the last 24 hours.\n"
        "- Begin with a sentence explaining that here is 5 new papers from arXiv on artificial intelligence that are worth a look.\n"
        "- Dedicate one short paragraph per paper starting with the ranking number (1. / 2. / 3. / etc.) followed with the exact paper title enclosed in double quotes (e.g., \"Attention Is All You Need\"),"
        " followed by the key idea, why it matters, and a concise practical takeaway.\n"
        "- Under each paragraph, add the link sentence formatted as"
        " Lien: <url>.\n"
        f"- After the five paragraphs, append a one-sentence disclaimer that the content was generated"
        " by a LangChain agentic workflow and link to the open-source code at "
        f"{REPO_URL}.\n"
        "Constraints: stay factual, avoid marketing buzzwords and filler, do not use emojis, do not use"
        " any Markdown (no bullets, numbered lists, bold, headings, or code), and write plain text"
        " sentences separated only by blank lines (with the link sentence on its own line).\n"
        "Keep the tone approachable for a broad audience: emphasize what each paper enables in practice, how it impacts teams or products, and the concrete benefits readers could expect. Use plain-language analogies, focus on use cases, and avoid dense jargon or architecture deep dives.\n"
        "Use the following research brief as context (do not list raw metadata, turn it into prose):\n"
        f"{brief}"
    )
