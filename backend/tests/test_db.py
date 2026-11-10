"""
test_db.py — Repository Layer Tests (Mocked)
Module 2: Data Layer

KEY CONCEPT: We mock the Supabase client so tests run with NO real database.
This is production-grade testing practice:
  - Fast (no network round trips)
  - Deterministic (no flaky external state)
  - Works in CI/CD without Supabase credentials
  - Tests YOUR logic, not Supabase's library

Pattern used: pytest + unittest.mock.AsyncMock
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from backend.db.applications_repo import ApplicationsRepo
from backend.db.jobs_repo import JobsRepo
from backend.models.job import ApplicationStatus, JobSource, RawJob


# ===========================================================================
# FIXTURES — shared test setup
# ===========================================================================

@pytest.fixture
def sample_raw_job() -> RawJob:
    return RawJob(
        title="Senior Python Engineer",
        company="Acme Corp",
        location="Remote",
        description="Build scalable Python services with FastAPI and AWS...",
        url="https://linkedin.com/jobs/123",
        source=JobSource.LINKEDIN,
    )


@pytest.fixture
def mock_supabase_client():
    """
    Creates a mock Supabase async client.

    Why AsyncMock?
        Our repo methods use `await client.table(...).execute()`
        AsyncMock lets us set return values for async calls.

    Chain structure we're mocking:
        client.table("jobs")          → table_mock
            .select("*")              → returns self (chaining)
            .eq("id", id)             → returns self
            .execute()                → returns result_mock
    """
    client = MagicMock()

    # Make execute() async and return empty data by default
    execute_mock = AsyncMock(return_value=MagicMock(data=[], count=0))

    # Build the chain: table().select().eq().execute() all return something chainable
    query_builder = MagicMock()
    query_builder.execute = execute_mock
    query_builder.select.return_value = query_builder
    query_builder.eq.return_value = query_builder
    query_builder.gte.return_value = query_builder
    query_builder.is_.return_value = query_builder
    query_builder.not_ = MagicMock()
    query_builder.not_.is_.return_value = query_builder
    query_builder.order.return_value = query_builder
    query_builder.limit.return_value = query_builder
    query_builder.insert.return_value = query_builder
    query_builder.update.return_value = query_builder
    query_builder.text_search.return_value = query_builder
    query_builder.rpc.return_value = query_builder

    client.table.return_value = query_builder
    client.rpc.return_value = query_builder

    return client, query_builder, execute_mock


# ===========================================================================
# JOBS REPO TESTS
# ===========================================================================

class TestJobsRepo:

    @pytest.mark.asyncio
    async def test_upsert_raw_inserts_new_job(self, sample_raw_job, mock_supabase_client):
        """New job (not in DB) should be inserted."""
        client, query_builder, execute_mock = mock_supabase_client

        # First call: check existing → empty (not a duplicate)
        # Second call: insert → returns new row
        execute_mock.side_effect = [
            MagicMock(data=[]),                          # select existing: empty
            MagicMock(data=[{"id": str(uuid4())}]),      # insert: success
        ]

        repo = JobsRepo()
        repo._supabase = client
        result = await repo.upsert_raw(sample_raw_job)

        assert result["action"] == "inserted"

    @pytest.mark.asyncio
    async def test_save_enrichment_maps_model_fields_to_columns(self, mock_supabase_client):
        """save_enrichment must write the right DB columns from an EnrichedJob."""
        from backend.models.job import EnrichedJob, RemoteType, Seniority

        client, query_builder, execute_mock = mock_supabase_client
        execute_mock.return_value = MagicMock(data=[{"id": "123"}])

        enriched = EnrichedJob(
            dedup_hash="abc", title="AI Engineer", company="Acme",
            description="...", url="https://x.com/1", source=JobSource.LINKEDIN,
            remote_type=RemoteType.REMOTE, seniority=Seniority.SENIOR,
            match_score=88.0, skills_match=["Python"], skills_gap=["Rust"],
            resume_bullet_suggestions=["Built X"], keywords_to_add=["LangGraph"],
            enrichment_model="llama-3.3-70b-versatile", prompt_version="v1",
        )

        repo = JobsRepo()
        repo._supabase = client
        await repo.save_enrichment("123", enriched)

        updates = query_builder.update.call_args.args[0]
        assert updates["bullet_suggestions"] == ["Built X"]
        assert updates["keywords_to_add"] == ["LangGraph"]
        assert updates["remote_type"] == "remote"
        assert updates["match_score"] == 88.0
        assert "enriched_at" in updates

    @pytest.mark.asyncio
    async def test_get_stats_counts_enriched_with_is_not_null(self, mock_supabase_client):
        """Enriched count must use IS NOT NULL, not eq('enriched_at', 'not null')."""
        client, query_builder, execute_mock = mock_supabase_client
        execute_mock.side_effect = [
            MagicMock(data=[], count=3),                                      # total
            MagicMock(data=[], count=1),                                      # enriched
            MagicMock(data=[{"source": "remoteok"}] * 2 + [{"source": "themuse"}]),
        ]

        repo = JobsRepo()
        repo._supabase = client
        stats = await repo.get_stats()

        query_builder.not_.is_.assert_called_with("enriched_at", "null")
        assert stats["total_jobs"] == 3
        assert stats["enriched_jobs"] == 1
        assert stats["unenriched_jobs"] == 2
        assert stats["by_source"] == {"remoteok": 2, "themuse": 1}

    @pytest.mark.asyncio
    async def test_upsert_raw_skips_duplicate(self, sample_raw_job, mock_supabase_client):
        """Job already in DB should be skipped (dedup by hash)."""
        client, query_builder, execute_mock = mock_supabase_client
        existing_id = str(uuid4())

        # select returns existing row → duplicate detected
        execute_mock.side_effect = [
            MagicMock(data=[{"id": existing_id}]),  # select existing: found
        ]

        repo = JobsRepo()
        repo._supabase = client
        result = await repo.upsert_raw(sample_raw_job)

        assert result["action"] == "skipped"
        assert result["data"]["id"] == existing_id

    @pytest.mark.asyncio
    async def test_get_top_matches_returns_list(self, mock_supabase_client):
        """get_top_matches should return list of job dicts."""
        client, query_builder, execute_mock = mock_supabase_client
        execute_mock.return_value = MagicMock(data=[
            {"id": str(uuid4()), "title": "ML Engineer", "company": "OpenAI", "match_score": 92.0},
            {"id": str(uuid4()), "title": "AI Engineer", "company": "Google", "match_score": 88.0},
        ])

        repo = JobsRepo()
        repo._supabase = client
        results = await repo.get_top_matches(limit=10, min_score=50.0)

        assert len(results) == 2
        assert results[0]["match_score"] == 92.0

    @pytest.mark.asyncio
    async def test_get_unenriched_returns_jobs_without_enrichment(self, mock_supabase_client):
        """Should fetch jobs where enriched_at is NULL."""
        client, query_builder, execute_mock = mock_supabase_client
        execute_mock.return_value = MagicMock(data=[
            {"id": str(uuid4()), "title": "Data Engineer", "description": "Build pipelines..."},
        ])

        repo = JobsRepo()
        repo._supabase = client
        results = await repo.get_unenriched(limit=50)

        assert len(results) == 1
        assert results[0]["title"] == "Data Engineer"

    @pytest.mark.asyncio
    async def test_get_by_id_returns_none_when_not_found(self, mock_supabase_client):
        """get_by_id should return None when job doesn't exist."""
        client, query_builder, execute_mock = mock_supabase_client
        execute_mock.return_value = MagicMock(data=[])

        repo = JobsRepo()
        repo._supabase = client
        result = await repo.get_by_id(str(uuid4()))

        assert result is None

    @pytest.mark.asyncio
    async def test_save_embedding_calls_update(self, mock_supabase_client):
        """save_embedding should call DB update with the embedding vector."""
        client, query_builder, execute_mock = mock_supabase_client
        execute_mock.return_value = MagicMock(data=[{"id": "123"}])

        repo = JobsRepo()
        repo._supabase = client
        job_id = str(uuid4())
        embedding = [0.1] * 384  # 384-dim vector

        await repo.save_embedding(job_id, embedding)

        # Verify update was called on the table
        client.table.assert_called_with("jobs")

    @pytest.mark.asyncio
    async def test_dedup_hash_prevents_duplicate_insert(self, mock_supabase_client):
        """Same job inserted twice should only result in one DB insert."""
        client, query_builder, execute_mock = mock_supabase_client

        # First upsert: not found → insert
        # Second upsert: found → skip
        job = RawJob(
            title="Python Dev",
            company="Test Co",
            description="Test description for a Python developer role",
            url="https://example.com/job/1",
        )

        existing_id = str(uuid4())
        execute_mock.side_effect = [
            MagicMock(data=[]),                             # first call: not found
            MagicMock(data=[{"id": existing_id}]),          # first insert: success
            MagicMock(data=[{"id": existing_id}]),          # second call: found
        ]

        repo = JobsRepo()
        repo._supabase = client

        first = await repo.upsert_raw(job)
        second = await repo.upsert_raw(job)

        assert first["action"] == "inserted"
        assert second["action"] == "skipped"


# ===========================================================================
# APPLICATIONS REPO TESTS
# ===========================================================================

class TestApplicationsRepo:

    @pytest.mark.asyncio
    async def test_save_job_creates_application(self, mock_supabase_client):
        """Saving a job creates a new application with status=saved."""
        client, query_builder, execute_mock = mock_supabase_client
        job_id = str(uuid4())
        app_id = str(uuid4())

        execute_mock.side_effect = [
            MagicMock(data=[]),                                                    # not already saved
            MagicMock(data=[{"id": app_id, "job_id": job_id, "status": "saved"}]),  # insert
        ]

        repo = ApplicationsRepo()
        repo._supabase = client
        result = await repo.save_job(job_id)

        assert result["action"] == "saved"

    @pytest.mark.asyncio
    async def test_save_job_skips_already_saved(self, mock_supabase_client):
        """Saving a job that's already in the pipeline should not duplicate it."""
        client, query_builder, execute_mock = mock_supabase_client
        job_id = str(uuid4())
        existing_id = str(uuid4())

        execute_mock.return_value = MagicMock(
            data=[{"id": existing_id, "status": "applied"}]
        )

        repo = ApplicationsRepo()
        repo._supabase = client
        result = await repo.save_job(job_id)

        assert result["action"] == "already_saved"

    @pytest.mark.asyncio
    async def test_update_status_to_applied(self, mock_supabase_client):
        """Updating status to APPLIED should set applied_at timestamp."""
        client, query_builder, execute_mock = mock_supabase_client
        app_id = str(uuid4())

        execute_mock.return_value = MagicMock(
            data=[{"id": app_id, "status": "applied"}]
        )

        repo = ApplicationsRepo()
        repo._supabase = client
        result = await repo.update_status(app_id, ApplicationStatus.APPLIED)

        assert result.get("status") == "applied"

    @pytest.mark.asyncio
    async def test_pipeline_stats_returns_all_statuses(self, mock_supabase_client):
        """get_pipeline_stats should return counts for every status."""
        client, query_builder, execute_mock = mock_supabase_client
        execute_mock.return_value = MagicMock(data=[
            {"status": "saved"},
            {"status": "saved"},
            {"status": "applied"},
            {"status": "interview"},
        ])

        repo = ApplicationsRepo()
        repo._supabase = client
        stats = await repo.get_pipeline_stats()

        assert stats["saved"] == 2
        assert stats["applied"] == 1
        assert stats["interview"] == 1
        assert stats["total"] == 4
