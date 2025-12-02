import os
import json
from pathlib import Path

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def paper_id_from_url(url: str) -> str:
    """Extrait l'ID du papier depuis l'URL arXiv."""
    return url.rstrip("/").split("/")[-1]

def cache_path(paper_id: str) -> Path:
    return CACHE_DIR / f"{paper_id}.json"

def load_cache(paper_id: str) -> dict | None:
    path = cache_path(paper_id)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None

def save_cache(paper_id: str, data: dict):
    path = cache_path(paper_id)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
