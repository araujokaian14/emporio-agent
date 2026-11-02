"""Central configuration. Every value can be overridden through environment variables (.env)."""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))

# Any OpenAI-compatible endpoint works: Ollama (default), OpenAI, Groq, vLLM...
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")  # Ollama ignores the key
LLM_MODEL = os.getenv("LLM_MODEL", "granite4.1:8b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# The store is in Campo Grande/MS (UTC-4), not in São Paulo.
STORE_TZ = ZoneInfo(os.getenv("STORE_TZ", "America/Campo_Grande"))


def now() -> datetime:
    """Current time at the store. Set FAKE_NOW=2026-11-02T10:00 to simulate another moment."""
    fake = os.getenv("FAKE_NOW")
    if fake:
        return datetime.fromisoformat(fake).replace(tzinfo=STORE_TZ)
    return datetime.now(STORE_TZ)


def today() -> date:
    return now().date()
