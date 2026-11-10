"""
base_repo.py — Base Repository Pattern
Module 2: Data Layer

Design Pattern: Repository Pattern
- Abstracts all database access behind clean methods
- Makes unit testing easy (swap real DB for mock)
- One place to change if you switch from Supabase to Postgres directly
- Every repo inherits from BaseRepo for shared connection logic
"""
from __future__ import annotations

import logging
from typing import Any

from supabase import AsyncClient, acreate_client

from backend.config import get_settings

logger = logging.getLogger(__name__)


class BaseRepo:
    """
    Base class for all repository classes.

    Usage:
        class JobsRepo(BaseRepo):
            TABLE = "jobs"

            async def get_by_id(self, job_id: str):
                client = await self._client()
                result = await client.table(self.TABLE).select("*").eq("id", job_id).execute()
                return result.data[0] if result.data else None

    Why async?
        Supabase Python SDK v2 supports async clients.
        FastAPI is async, APScheduler supports async jobs.
        Async = no blocking I/O while waiting for DB responses.
    """

    TABLE: str = ""  # subclasses must set this

    def __init__(self) -> None:
        self._settings = get_settings()
        self._supabase: AsyncClient | None = None

    async def _client(self) -> AsyncClient:
        """
        Lazy singleton: creates the Supabase client once, reuses it.

        Why lazy?
            - Config isn't loaded until first use
            - Tests can override settings before first DB call
            - No connection at import time (safe for unit tests)

        Why service_role key?
            - Bypasses Row Level Security (RLS)
            - Only used in backend (never exposed to frontend)
            - anon key would hit RLS policies and be blocked
        """
        if self._supabase is None:
            self._supabase = await acreate_client(
                supabase_url=self._settings.supabase_url,
                supabase_key=self._settings.supabase_service_role_key,
            )
        return self._supabase

    async def _execute_query(self, query: Any) -> list[dict]:
        """
        Executes a Supabase query and returns data list.
        Centralizes error handling and logging.
        """
        try:
            result = await query.execute()
            return result.data or []
        except Exception as e:
            logger.error(f"[{self.TABLE}] DB query failed: {e}")
            raise

    async def count(self, filters: dict | None = None) -> int:
        """Count rows, optionally with eq filters."""
        client = await self._client()
        q = client.table(self.TABLE).select("id", count="exact")
        if filters:
            for col, val in filters.items():
                q = q.eq(col, val)
        result = await q.execute()
        return result.count or 0
