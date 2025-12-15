from pathlib import Path

from pydantic import BaseSettings, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
DEFAULT_CATEGORIES = ["cs.CL", "cs.AI", "cs.IR", "cs.MA"]
REPO_URL = "https://github.com/eric-houzelle/arxiv-agent"
LINKEDIN_CHARACTER_LIMIT = 2500
CRITERIA_PROMPT_FILES = [
    ("originalite", "Originality", "originality.md"),
    ("impact", "Technical impact", "impact.md"),
    ("repro", "Reproducibility", "repro.md"),
    ("potentiel", "Short-term potential", "potential.md"),
]


class AppSettings(BaseSettings):
    """Configuration typée de l'application, chargée depuis l'environnement."""

    linkedin_post_language: str = Field(
        "fr", alias="LINKEDIN_POST_LANGUAGE", description="Langue du post LinkedIn"
    )
    linkedin_post_temperature: float = Field(
        0.4,
        alias="LINKEDIN_POST_TEMPERATURE",
        description="Température de génération du post LinkedIn",
    )

    model_config = {"extra": "ignore"}


_settings = AppSettings()  # instancié une seule fois au démarrage


def linkedin_language() -> str:
    """Retourne la langue configurée pour les posts LinkedIn."""
    return _settings.linkedin_post_language


def linkedin_temperature() -> float:
    """Retourne la température configurée pour les posts LinkedIn."""
    return _settings.linkedin_post_temperature
