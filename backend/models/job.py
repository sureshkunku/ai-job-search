"""
Job model — the core data structure everything else is built on.

Learn: Pydantic v2 — Field validators, computed fields, enums, UUID,
       datetime handling, and how to make models self-documenting.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl, computed_field, field_validator


# ── Enums ────────────────────────────────────────────────────────────────────
# Always use Enum for fixed sets of values — never raw strings.
# This prevents typos and makes IDE autocomplete work.

class RemoteType(str, Enum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    UNKNOWN = "unknown"


class Seniority(str, Enum):
    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    STAFF = "staff"
    PRINCIPAL = "principal"
    UNKNOWN = "unknown"


class JobSource(str, Enum):
    LINKEDIN = "linkedin"
    INDEED = "indeed"
    DICE = "dice"
    GLASSDOOR = "glassdoor"
    COMPANY_SITE = "company_site"
    UNKNOWN = "unknown"


class ApplicationStatus(str, Enum):
    SAVED = "saved"
    APPLIED = "applied"
    SCREENING = "screening"
    INTERVIEW = "interview"
    OFFER = "offer"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


# ── Raw Job (from scraper — before AI enrichment) ────────────────────────────

class RawJob(BaseModel):
    """
    What comes directly out of the scraper — minimal, unvalidated.
    We separate raw from enriched because scraping is unreliable.
    Some fields may be None until AI enrichment fills them.
    """
    title: str
    company: str
    location: str | None = None
    description: str
    url: str                              # unique identifier from source
    source: JobSource = JobSource.UNKNOWN
    posted_at: datetime | None = None
    salary_raw: str | None = None         # "120k-150k" — unparsed string from listing

    @field_validator("title", "company", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """Remove accidental whitespace from scraped strings."""
        return v.strip() if isinstance(v, str) else v

    @computed_field  # type: ignore[misc]
    @property
    def dedup_hash(self) -> str:
        """
        SHA-256 of title + company + location.
        Used to detect duplicate jobs across sources.
        Same job posted on LinkedIn AND Indeed gets the same hash.
        """
        raw = f"{self.title.lower()}{self.company.lower()}{(self.location or '').lower()}"
        return hashlib.sha256(raw.encode()).hexdigest()


# ── Enriched Job (after AI agents process it) ────────────────────────────────

class EnrichedJob(BaseModel):
    """
    Full job record after AI enrichment pipeline.
    Every AI-extracted field is Optional — agents can fail gracefully.
    """
    # Identity
    id: UUID = Field(default_factory=uuid4)
    dedup_hash: str                        # from RawJob — prevents duplicates

    # Core fields (from scraper)
    title: str
    company: str
    location: str | None = None
    description: str
    url: str
    source: JobSource
    posted_at: datetime | None = None
    scraped_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Salary (parsed by AI from salary_raw string)
    salary_min: int | None = None          # in USD annual
    salary_max: int | None = None
    salary_currency: str = "USD"

    # AI-extracted fields (Enrichment Agent)
    required_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    experience_years_min: int | None = None
    experience_years_max: int | None = None
    remote_type: RemoteType = RemoteType.UNKNOWN
    seniority: Seniority = Seniority.UNKNOWN
    tech_stack: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)

    # AI scoring fields (Scoring Agent)
    match_score: float | None = None       # 0.0 - 100.0
    match_reasoning: str | None = None     # why this score
    skills_gap: list[str] = Field(default_factory=list)   # skills you're missing
    skills_match: list[str] = Field(default_factory=list) # skills you have

    # Resume tailor suggestions (Resume Tailor Agent)
    resume_bullet_suggestions: list[str] = Field(default_factory=list)
    keywords_to_add: list[str] = Field(default_factory=list)  # ATS keywords missing from resume

    # Embedding (set after vectorization step)
    embedding: list[float] | None = None   # vector for semantic search

    # Metadata
    enriched_at: datetime | None = None
    enrichment_model: str | None = None    # track which LLM version enriched this
    prompt_version: str | None = None      # track which prompt version was used

    @computed_field  # type: ignore[misc]
    @property
    def salary_display(self) -> str:
        """Human-readable salary range, e.g. '$120k – $150k'"""
        if self.salary_min and self.salary_max:
            return f"${self.salary_min // 1000}k – ${self.salary_max // 1000}k"
        if self.salary_min:
            return f"${self.salary_min // 1000}k+"
        return "Not specified"

    @computed_field  # type: ignore[misc]
    @property
    def is_good_match(self) -> bool:
        """Quick boolean check — did AI score this above threshold?"""
        return self.match_score is not None and self.match_score >= 75.0

    @field_validator("match_score")
    @classmethod
    def validate_score_range(cls, v: float | None) -> float | None:
        """Scores must be 0-100. Catch LLM hallucinations that return 110%."""
        if v is not None and not (0.0 <= v <= 100.0):
            raise ValueError(f"match_score must be 0-100, got {v}")
        return v

    def to_search_text(self) -> str:
        """
        Combined text used for BM25 indexing.
        We weight title and skills higher by repeating them.
        """
        parts = [
            self.title,
            self.title,                        # repeated = higher weight
            self.company,
            " ".join(self.required_skills),
            " ".join(self.required_skills),    # repeated
            " ".join(self.tech_stack),
            self.description[:500],            # first 500 chars of JD
        ]
        return " ".join(filter(None, parts))


# ── API Response Models (what the frontend sees) ─────────────────────────────

class JobSummary(BaseModel):
    """
    Lightweight version for job listing pages — no description blob.
    Use this for lists; use EnrichedJob for the detail view.
    """
    id: UUID
    title: str
    company: str
    location: str | None
    remote_type: RemoteType
    seniority: Seniority
    salary_display: str
    match_score: float | None
    is_good_match: bool
    source: JobSource
    posted_at: datetime | None
    top_matching_skills: list[str] = Field(default_factory=list)
    skills_gap_count: int = 0


class JobSearchRequest(BaseModel):
    """Request body for /search endpoint."""
    query: str = Field(..., min_length=3, max_length=500)
    remote_only: bool = False
    min_salary: int | None = None
    min_match_score: float | None = None
    seniority: list[Seniority] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=50)


class JobSearchResponse(BaseModel):
    """Response from /search endpoint."""
    query: str
    results: list[JobSummary]
    total_found: int
    search_type: str   # "hybrid" | "semantic" | "keyword"
    latency_ms: float
