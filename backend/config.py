"""
Config management with pydantic-settings.

Learn:
  - pydantic-settings reads from .env automatically
  - Settings are VALIDATED at startup — app won't run with bad config
  - Use @lru_cache so we read .env once, not on every import
  - Type hints enforce correct types (int, bool, HttpUrl etc.)
  - Never use os.environ.get() scattered through the codebase
"""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All configuration in one place.
    pydantic-settings reads these from environment variables or .env file.
    Variable names are case-insensitive (GROQ_API_KEY == groq_api_key).
    """

    model_config = SettingsConfigDict(
        env_file=".env",           # read from .env file
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",            # ignore unknown env vars (don't crash)
        populate_by_name=True,     # allow Settings(field_name=...) alongside aliases
    )

    # ── App ───────────────────────────────────────────────────────────────
    app_env: str = Field(default="development")
    log_level: str = Field(default="DEBUG")
    api_secret_key: str = Field(default="dev-secret-change-in-prod")

    @field_validator("app_env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"app_env must be one of {allowed}")
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    # ── LLM ───────────────────────────────────────────────────────────────
    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="llama-3.3-70b-versatile")  # free, fast, smart
    groq_max_tokens: int = Field(default=2048)
    groq_temperature: float = Field(default=0.1)  # low = more deterministic for structured output

    # ── Database ──────────────────────────────────────────────────────────
    supabase_url: str = Field(default="")
    # Supabase renamed its keys: anon → publishable, service_role → secret.
    # Accept either name from .env so old and new projects both work.
    supabase_anon_key: str = Field(
        default="",
        validation_alias=AliasChoices("supabase_anon_key", "supabase_publishable_key"),
    )
    supabase_service_role_key: str = Field(
        default="",
        validation_alias=AliasChoices("supabase_service_role_key", "supabase_secret_key"),
    )

    # ── Scraping ──────────────────────────────────────────────────────────
    apify_api_token: str = Field(default="")
    serpapi_key: str = Field(default="")
    scrape_interval_hours: int = Field(default=6)
    max_jobs_per_run: int = Field(default=200)

    # Search keywords for scraping (your target roles)
    job_search_keywords: list[str] = Field(default=[
        "Senior Platform Engineer Python FastAPI",
        "AI Engineer LangChain AWS",
        "Senior Python Engineer FastAPI",
        "Data Engineer Databricks AWS",
        "ML Platform Engineer",
        "AI Platform Engineer",
    ])
    job_search_locations: list[str] = Field(default=[
        "Remote",
        "Dallas TX",
        "Irving TX",
        "Austin TX",
    ])

    # ── Scraper behaviour (Module 3) ──────────────────────────────────────
    # A job is kept only if one of these appears in its title or tags.
    # WHY titles+tags and not the description: almost every tech JD mentions
    # "Python" somewhere, so matching the description lets in sales and
    # frontend roles. Matching is whole-word ("ml" won't match "html").
    scraper_title_keywords: list[str] = Field(default=[
        "python", "ai", "ml", "machine learning", "llm", "genai",
        "generative ai", "data engineer", "platform engineer", "mlops",
        "backend", "back-end", "fastapi", "data scientist",
    ])
    # Sent to sources that have a search endpoint (Himalayas).
    scraper_search_terms: list[str] = Field(default=[
        "python", "ai engineer", "machine learning", "data engineer",
        "platform engineer", "llm",
    ])
    scraper_max_pages: int = Field(default=3)              # per source / search term
    scraper_min_interval_seconds: float = Field(default=1.0)  # politeness gap between requests
    scraper_timeout_seconds: float = Field(default=30.0)
    # Honest, identifiable User-Agent — never pretend to be a browser.
    scraper_user_agent: str = Field(
        default="ai-job-search/0.1 (personal job aggregator; learning project)"
    )
    # The Muse blocks anonymous automated requests; the scraper only runs
    # when you register a free key at https://www.themuse.com/developers/api/v2
    themuse_api_key: str = Field(default="")

    # ── Observability ─────────────────────────────────────────────────────
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    # ── Email ─────────────────────────────────────────────────────────────
    sendgrid_api_key: str = Field(default="")
    email_from: str = Field(default="noreply@example.com")
    email_to: str = Field(default="")
    match_score_threshold: float = Field(default=75.0)

    # ── Embeddings ────────────────────────────────────────────────────────
    # all-MiniLM-L6-v2: fast, free, 384 dimensions, good for semantic search
    # all-mpnet-base-v2: slower, free, 768 dimensions, more accurate
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_dimensions: int = Field(default=384)   # must match vector(384) in setup_db.sql
    embed_batch_size: int = Field(default=32)         # texts per model forward pass

    # ── Chunking + search (Module 4) ──────────────────────────────────────
    # MiniLM reads at most ~256 word-pieces; ~200 words stays inside that.
    chunk_max_words: int = Field(default=200)
    chunk_overlap_words: int = Field(default=30)      # context carried across a split
    # Cross-encoder: reads (query, job) TOGETHER → slower but far more accurate.
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L6-v2")
    search_candidates: int = Field(default=20)        # per retriever, and how many get reranked
    rrf_k: int = Field(default=60)                    # RRF damping constant (60 = paper default)
    # Resume chunks + vectors cached here (gitignored — it contains your resume).
    resume_index_path: str = Field(default="./data/resume_index.json")

    # ── Resume ────────────────────────────────────────────────────────────
    resume_file_path: str = Field(default="./data/resume.txt")

    @property
    def resume_text(self) -> str:
        """Load resume text from a .txt or .pdf file."""
        path = Path(self.resume_file_path)
        if path.exists():
            if path.suffix.lower() == ".pdf":
                from pypdf import PdfReader  # lazy import: only needed for PDFs

                reader = PdfReader(path)
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            return path.read_text(encoding="utf-8")
        # Return placeholder so app doesn't crash during development
        return "Senior Platform Engineer with 12 years experience. Python, FastAPI, AWS, Databricks, LangChain, LangGraph."

    # ── Scoring thresholds ────────────────────────────────────────────────
    score_excellent: float = Field(default=90.0)
    score_good: float = Field(default=75.0)
    score_stretch: float = Field(default=50.0)

    # ── Cost guardrails (MLOps — stop runaway LLM costs) ─────────────────
    max_tokens_per_job: int = Field(default=4000)
    max_cost_per_job_usd: float = Field(default=0.01)   # $0.01 per job max
    max_daily_cost_usd: float = Field(default=1.00)     # $1/day hard limit
    max_agent_iterations: int = Field(default=10)        # LangGraph loop guard


@lru_cache  # Read .env once at startup, reuse everywhere
def get_settings() -> Settings:
    """
    Use this everywhere instead of creating Settings() directly.

    Usage:
        from backend.config import get_settings
        settings = get_settings()
        print(settings.groq_api_key)

    Why lru_cache?
        - Settings() reads from disk — do it once
        - In tests, you can mock get_settings() to inject test values
    """
    return Settings()
