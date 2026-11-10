"""
Enrichment output models — structured output from LangGraph agents.

Learn: How to use Pydantic as the OUTPUT SCHEMA for LLM calls.
       The LLM is forced to return JSON matching these exact shapes.
       If it doesn't, Pydantic raises a ValidationError — caught and retried.
"""

from pydantic import BaseModel, Field, field_validator


class EnrichmentOutput(BaseModel):
    """
    Output schema for the Enrichment Agent.
    LLM MUST return JSON matching this exactly — Pydantic enforces it.
    """
    required_skills: list[str] = Field(
        description="Skills explicitly required in the job description",
        default_factory=list,
    )
    nice_to_have_skills: list[str] = Field(
        description="Skills listed as preferred or nice-to-have",
        default_factory=list,
    )
    tech_stack: list[str] = Field(
        description="Technologies, frameworks, tools mentioned anywhere",
        default_factory=list,
    )
    responsibilities: list[str] = Field(
        description="Key job responsibilities, max 5 bullet points",
        default_factory=list,
        max_length=5,
    )
    experience_years_min: int | None = Field(
        default=None,
        description="Minimum years of experience required",
        ge=0, le=30,
    )
    experience_years_max: int | None = Field(
        default=None,
        description="Maximum years of experience mentioned",
        ge=0, le=30,
    )
    remote_type: str = Field(
        default="unknown",
        description="One of: remote, hybrid, onsite, unknown",
    )
    seniority: str = Field(
        default="unknown",
        description="One of: intern, junior, mid, senior, staff, principal, unknown",
    )
    salary_min: int | None = Field(
        default=None,
        description="Minimum salary in USD annual, null if not mentioned",
    )
    salary_max: int | None = Field(
        default=None,
        description="Maximum salary in USD annual, null if not mentioned",
    )

    @field_validator("remote_type")
    @classmethod
    def validate_remote_type(cls, v: str) -> str:
        allowed = {"remote", "hybrid", "onsite", "unknown"}
        return v.lower() if v.lower() in allowed else "unknown"

    @field_validator("seniority")
    @classmethod
    def validate_seniority(cls, v: str) -> str:
        allowed = {"intern", "junior", "mid", "senior", "staff", "principal", "unknown"}
        return v.lower() if v.lower() in allowed else "unknown"


class ScoringOutput(BaseModel):
    """
    Output schema for the Scoring Agent.
    Scores your resume fit against the job description.
    """
    match_score: float = Field(
        description="Overall match score 0-100. 90+ = excellent, 70-89 = good, 50-69 = stretch, <50 = poor fit",
        ge=0.0, le=100.0,
    )
    match_reasoning: str = Field(
        description="2-3 sentence explanation of why this score was given",
        max_length=500,
    )
    skills_match: list[str] = Field(
        description="Required skills from the job that the candidate HAS",
        default_factory=list,
    )
    skills_gap: list[str] = Field(
        description="Required skills from the job that the candidate LACKS",
        default_factory=list,
    )
    experience_fit: str = Field(
        description="One of: overqualified, good_fit, underqualified, unknown",
        default="unknown",
    )
    standout_strengths: list[str] = Field(
        description="Top 3 things in the resume that make this candidate stand out for THIS job",
        default_factory=list,
        max_length=3,
    )


class ResumeTailorOutput(BaseModel):
    """
    Output schema for the Resume Tailor Agent.
    Suggests specific bullet points to add/update on your resume for this job.
    """
    bullet_suggestions: list[str] = Field(
        description="3 specific resume bullet points tailored to this job. Use metrics where possible.",
        min_length=1,
        max_length=3,
    )
    keywords_to_add: list[str] = Field(
        description="Keywords from the JD missing from resume that ATS systems will scan for",
        default_factory=list,
        max_length=10,
    )
    summary_suggestion: str | None = Field(
        default=None,
        description="Suggested one-line resume summary tweak for this specific role",
        max_length=200,
    )


class FullEnrichmentResult(BaseModel):
    """
    Combined output of all three agents.
    This is what gets stored in the database per job.
    """
    job_id: str
    enrichment: EnrichmentOutput
    scoring: ScoringOutput
    tailor: ResumeTailorOutput

    # MLOps tracking — always record which model/prompt produced this
    llm_model: str
    prompt_version: str
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    processing_time_ms: float = 0.0
