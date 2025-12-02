import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
DEFAULT_CATEGORIES = ["cs.CL", "cs.AI", "cs.IR", "cs.MA"]
REPO_URL = "https://github.com/eric-houzelle/arxiv-agent"
CRITERIA_PROMPT_FILES = [
    ("originalite", "Originality", "originality.md"),
    ("impact", "Technical impact", "impact.md"),
    ("repro", "Reproducibility", "repro.md"),
    ("potentiel", "Short-term potential", "potential.md"),
]


def linkedin_language() -> str:
    return os.getenv("LINKEDIN_POST_LANGUAGE", "fr")


def linkedin_temperature() -> float:
    value = os.getenv("LINKEDIN_POST_TEMPERATURE")
    try:
        return float(value) if value is not None else 0.4
    except ValueError:
        return 0.4
