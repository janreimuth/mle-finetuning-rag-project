import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# Walks up from this file's folder until it finds a file named .venv.
load_dotenv(find_dotenv(".venv"))

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Commit of all-MiniLM-L6-v2 on the Hugging Face Hub, pinned for reproducibility.
DEFAULT_EMBEDDING_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


def _get(name: str, default: str = "") -> str:
    """Read an env var; empty values count as unset."""
    value = os.getenv(name, "").strip()
    return value or default


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    groq_api_key: str | None
    groq_model: str
    mistral_api_key: str | None
    mistral_model: str
    embedding_model: str
    embedding_revision: str | None
    top_k: int
    min_score: float
    vector_storage_path: Path | None  # None means in-memory index

    @property
    def llm_api_key(self) -> str | None:
        return {"groq": self.groq_api_key, "mistral": self.mistral_api_key}.get(self.llm_provider)

    @property
    def live_generation(self) -> bool:
        """True when the chosen provider has a key; otherwise run in dry mode."""
        return bool(self.llm_api_key)


def load_settings() -> Settings:
    model = _get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    # The pinned revision only applies to the default model.
    default_rev = DEFAULT_EMBEDDING_REVISION if model == DEFAULT_EMBEDDING_MODEL else ""
    storage = _get("VECTOR_STORAGE_PATH")

    settings = Settings(
        llm_provider=_get("LLM_PROVIDER", "groq").lower(),
        groq_api_key=_get("GROQ_API_KEY") or None,
        groq_model=_get("GROQ_MODEL", "openai/gpt-oss-20b"),
        mistral_api_key=_get("MISTRAL_API_KEY") or None,
        mistral_model=_get("MISTRAL_MODEL", "ministral-14b-latest"),
        embedding_model=model,
        embedding_revision=_get("EMBEDDING_REVISION", default_rev) or None,
        top_k=int(_get("TOP_K", "3")),
        min_score=float(_get("MIN_SCORE", "0.30")),
        vector_storage_path=Path(storage) if storage else None,
    )

    if settings.llm_provider not in ("groq", "mistral"):
        raise ValueError(f"LLM_PROVIDER must be 'groq' or 'mistral', got {settings.llm_provider!r}")
    if settings.top_k < 1:
        raise ValueError("TOP_K must be at least 1")
    if not 0 <= settings.min_score <= 1:
        raise ValueError("MIN_SCORE must be between 0 and 1")
    return settings


SETTINGS = load_settings()
