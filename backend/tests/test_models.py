"""
Module 1 tests — verify models and config work correctly.

Learn: Pytest fundamentals
  - test_ prefix = pytest picks it up automatically
  - assert = how you check results
  - pytest.raises = check that bad input raises the right error
  - No mocking needed here — pure unit tests, no external calls
"""

import pytest
from pydantic import ValidationError

from backend.models.job import (
    EnrichedJob,
    JobSource,
    RawJob,
    RemoteType,
    Seniority,
)
from backend.models.enrichment import EnrichmentOutput, ScoringOutput


# ── RawJob tests ─────────────────────────────────────────────────────────────

class TestRawJob:
    def test_creates_valid_raw_job(self):
        job = RawJob(
            title="Senior Python Engineer",
            company="Stripe",
            description="We need a Python expert...",
            url="https://stripe.com/jobs/123",
            source=JobSource.LINKEDIN,
        )
        assert job.title == "Senior Python Engineer"
        assert job.company == "Stripe"
        assert job.source == JobSource.LINKEDIN

    def test_strips_whitespace_from_title_and_company(self):
        """Scraped data often has extra whitespace."""
        job = RawJob(
            title="  Senior Engineer  ",
            company="  Stripe  ",
            description="...",
            url="https://example.com",
        )
        assert job.title == "Senior Engineer"
        assert job.company == "Stripe"

    def test_dedup_hash_is_deterministic(self):
        """Same job posted twice must produce same hash."""
        job1 = RawJob(title="Engineer", company="Stripe", description="...", url="url1")
        job2 = RawJob(title="Engineer", company="Stripe", description="...", url="url2")
        assert job1.dedup_hash == job2.dedup_hash

    def test_dedup_hash_differs_for_different_jobs(self):
        job1 = RawJob(title="Engineer", company="Stripe", description="...", url="url1")
        job2 = RawJob(title="Engineer", company="Plaid", description="...", url="url2")
        assert job1.dedup_hash != job2.dedup_hash

    def test_dedup_hash_case_insensitive(self):
        """'STRIPE' and 'stripe' should hash the same."""
        job1 = RawJob(title="Engineer", company="STRIPE", description="...", url="url1")
        job2 = RawJob(title="Engineer", company="stripe", description="...", url="url2")
        assert job1.dedup_hash == job2.dedup_hash


# ── EnrichedJob tests ─────────────────────────────────────────────────────────

class TestEnrichedJob:
    def test_salary_display_with_range(self):
        job = EnrichedJob(
            dedup_hash="abc",
            title="Engineer",
            company="Stripe",
            description="...",
            url="https://example.com",
            source=JobSource.LINKEDIN,
            salary_min=120_000,
            salary_max=160_000,
        )
        assert job.salary_display == "$120k – $160k"

    def test_salary_display_min_only(self):
        job = EnrichedJob(
            dedup_hash="abc",
            title="Engineer",
            company="Stripe",
            description="...",
            url="https://example.com",
            source=JobSource.LINKEDIN,
            salary_min=130_000,
        )
        assert job.salary_display == "$130k+"

    def test_salary_display_unknown(self):
        job = EnrichedJob(
            dedup_hash="abc",
            title="Engineer",
            company="Stripe",
            description="...",
            url="https://example.com",
            source=JobSource.LINKEDIN,
        )
        assert job.salary_display == "Not specified"

    def test_is_good_match_above_threshold(self):
        job = EnrichedJob(
            dedup_hash="abc", title="E", company="C",
            description="...", url="u", source=JobSource.LINKEDIN,
            match_score=85.0,
        )
        assert job.is_good_match is True

    def test_is_good_match_below_threshold(self):
        job = EnrichedJob(
            dedup_hash="abc", title="E", company="C",
            description="...", url="u", source=JobSource.LINKEDIN,
            match_score=60.0,
        )
        assert job.is_good_match is False

    def test_match_score_validation_rejects_over_100(self):
        """LLM might hallucinate a score of 110 — Pydantic catches it."""
        with pytest.raises(ValidationError) as exc_info:
            EnrichedJob(
                dedup_hash="abc", title="E", company="C",
                description="...", url="u", source=JobSource.LINKEDIN,
                match_score=110.0,   # invalid!
            )
        assert "match_score" in str(exc_info.value)

    def test_to_search_text_includes_title_and_skills(self):
        job = EnrichedJob(
            dedup_hash="abc", title="Python Engineer", company="Stripe",
            description="Build APIs", url="u", source=JobSource.LINKEDIN,
            required_skills=["FastAPI", "PostgreSQL"],
        )
        text = job.to_search_text()
        assert "Python Engineer" in text
        assert "FastAPI" in text
        assert "PostgreSQL" in text


# ── ScoringOutput tests ───────────────────────────────────────────────────────

class TestScoringOutput:
    def test_valid_score(self):
        score = ScoringOutput(
            match_score=87.5,
            match_reasoning="Strong Python and AWS skills match.",
            skills_match=["Python", "FastAPI"],
            skills_gap=["Kubernetes"],
        )
        assert score.match_score == 87.5

    def test_score_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            ScoringOutput(
                match_score=-5.0,
                match_reasoning="...",
            )

    def test_score_above_100_rejected(self):
        with pytest.raises(ValidationError):
            ScoringOutput(
                match_score=101.0,
                match_reasoning="...",
            )


# ── Config tests ──────────────────────────────────────────────────────────────

class TestConfig:
    def test_settings_loads_with_defaults(self):
        """Settings should load even without a .env file (uses defaults)."""
        from backend.config import Settings
        s = Settings()
        assert s.app_env == "development"
        assert s.groq_model == "llama-3.3-70b-versatile"
        assert s.embedding_dimensions == 384

    def test_invalid_app_env_raises(self):
        from backend.config import Settings
        with pytest.raises(Exception):
            Settings(app_env="invalid_env")

    def test_is_production_false_in_dev(self):
        from backend.config import Settings
        s = Settings(app_env="development")
        assert s.is_production is False

    def test_is_production_true_in_prod(self):
        from backend.config import Settings
        s = Settings(app_env="production")
        assert s.is_production is True
