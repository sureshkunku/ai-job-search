-- =============================================================
-- AI Job Search Aggregator — Supabase Schema
-- Module 2: Data Layer
-- Run this in Supabase SQL Editor (supabase.com → SQL Editor)
-- =============================================================

-- Enable pgvector extension (Supabase has it pre-installed)
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================
-- JOBS TABLE
-- Stores raw + enriched job postings
-- =============================================================
CREATE TABLE IF NOT EXISTS jobs (
    -- Identity
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dedup_hash      TEXT UNIQUE NOT NULL,   -- SHA256(title+company+url) prevents duplicates

    -- Raw scraped fields
    title           TEXT NOT NULL,
    company         TEXT NOT NULL,
    location        TEXT,
    description     TEXT NOT NULL,
    url             TEXT NOT NULL,
    source          TEXT NOT NULL DEFAULT 'unknown',  -- remoteok, themuse, arbeitnow, himalayas
    posted_at       TIMESTAMPTZ,
    scraped_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Enriched fields (filled by LangGraph agent)
    remote_type     TEXT,                   -- remote, hybrid, onsite, unknown
    seniority       TEXT,                   -- junior, mid, senior, etc.
    salary_min      INTEGER,
    salary_max      INTEGER,
    salary_currency TEXT DEFAULT 'USD',
    required_skills TEXT[],                 -- array of skill strings
    nice_to_have_skills TEXT[],
    benefits        TEXT[],

    -- AI scoring
    match_score     FLOAT,                  -- 0.0 - 100.0 vs your resume
    skills_match    TEXT[],                 -- skills you have
    skills_gap      TEXT[],                 -- skills you're missing

    -- Resume tailoring suggestions
    bullet_suggestions TEXT[],
    keywords_to_add    TEXT[],
    cover_letter_hook  TEXT,

    -- Vector embedding for semantic search (384 dims = all-MiniLM-L6-v2)
    embedding       vector(384),

    -- MLOps tracking
    enrichment_model   TEXT,               -- e.g. "llama-3.1-70b-versatile"
    prompt_version     TEXT,               -- e.g. "v1.2.0"
    enrichment_tokens  INTEGER,
    enrichment_cost    FLOAT,
    enriched_at        TIMESTAMPTZ,

    -- Timestamps
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =============================================================
-- APPLICATIONS TABLE
-- Tracks your job application pipeline
-- =============================================================
CREATE TABLE IF NOT EXISTS applications (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,

    -- Pipeline status
    status          TEXT NOT NULL DEFAULT 'saved',
    -- saved → applied → screening → interview → offer → rejected/withdrawn

    -- Application details
    applied_at      TIMESTAMPTZ,
    resume_version  TEXT,                  -- which resume variation you sent
    cover_letter    TEXT,

    -- Interview tracking
    interview_date  TIMESTAMPTZ,
    interview_notes TEXT,

    -- Outcome
    offer_amount    INTEGER,
    offer_currency  TEXT DEFAULT 'USD',
    rejection_reason TEXT,

    -- Personal notes
    notes           TEXT,
    priority        INTEGER DEFAULT 0,     -- 0=normal, 1=high, 2=dream job

    -- Timestamps
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =============================================================
-- SCRAPE_RUNS TABLE
-- Audit log — tracks every scrape for observability
-- =============================================================
CREATE TABLE IF NOT EXISTS scrape_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source          TEXT NOT NULL,         -- remoteok, themuse, arbeitnow, himalayas
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    jobs_found      INTEGER DEFAULT 0,
    jobs_new        INTEGER DEFAULT 0,
    jobs_duplicate  INTEGER DEFAULT 0,
    status          TEXT DEFAULT 'running',  -- running, success, failed
    error_message   TEXT
);

-- =============================================================
-- MIGRATIONS — columns added after the first version of this file.
-- ADD COLUMN IF NOT EXISTS makes re-running this script upgrade an
-- existing database without touching data that's already there.
-- =============================================================
-- Module 3: raw salary text from the job APIs, e.g. "USD 120,000–150,000 /year".
-- The Enrichment Agent (Module 6) parses it into salary_min / salary_max.
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS salary_raw TEXT;

-- =============================================================
-- INDEXES
-- =============================================================

-- Fast lookup by hash (dedup check on every new job)
CREATE INDEX IF NOT EXISTS idx_jobs_dedup_hash ON jobs(dedup_hash);

-- Filter by source (LinkedIn jobs, Indeed jobs, etc.)
CREATE INDEX IF NOT EXISTS idx_jobs_source ON jobs(source);

-- Filter by match score (show top matches)
CREATE INDEX IF NOT EXISTS idx_jobs_match_score ON jobs(match_score DESC NULLS LAST);

-- Filter by scraped date (recent jobs first)
CREATE INDEX IF NOT EXISTS idx_jobs_scraped_at ON jobs(scraped_at DESC);

-- Application pipeline queries
CREATE INDEX IF NOT EXISTS idx_applications_status ON applications(status);
CREATE INDEX IF NOT EXISTS idx_applications_job_id ON applications(job_id);

-- =============================================================
-- VECTOR INDEX (pgvector)
-- IVFFlat: faster queries, slight accuracy tradeoff
-- HNSW: more accurate, more memory — better for production
--
-- We use HNSW here (Supabase supports it, production-grade)
-- lists param for IVFFlat = rows/1000 (at least 100)
-- =============================================================
CREATE INDEX IF NOT EXISTS idx_jobs_embedding
    ON jobs USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
-- m=16: connections per node (16-64 range, higher = more accurate)
-- ef_construction=64: build-time accuracy (64-200 range)

-- =============================================================
-- AUTO-UPDATE updated_at TRIGGER
-- =============================================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- DROP ... IF EXISTS first so this whole script can be re-run safely
-- (CREATE TRIGGER has no IF NOT EXISTS and errors on a second run).
DROP TRIGGER IF EXISTS jobs_updated_at ON jobs;
CREATE TRIGGER jobs_updated_at
    BEFORE UPDATE ON jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS applications_updated_at ON applications;
CREATE TRIGGER applications_updated_at
    BEFORE UPDATE ON applications
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================
-- ROW-LEVEL SECURITY (RLS)
-- Supabase: enable RLS so anon key can't read your data
-- Only your service_role key (backend) can read/write
-- =============================================================
ALTER TABLE jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE applications ENABLE ROW LEVEL SECURITY;
ALTER TABLE scrape_runs ENABLE ROW LEVEL SECURITY;

-- Allow service_role full access (your backend uses this key)
-- (DROP first: CREATE POLICY also has no IF NOT EXISTS)
DROP POLICY IF EXISTS "service_role_all" ON jobs;
CREATE POLICY "service_role_all" ON jobs
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "service_role_all" ON applications;
CREATE POLICY "service_role_all" ON applications
    FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "service_role_all" ON scrape_runs;
CREATE POLICY "service_role_all" ON scrape_runs
    FOR ALL USING (auth.role() = 'service_role');

-- =============================================================
-- HELPER FUNCTION: semantic search with metadata filters
-- Called by JobsRepo.semantic_search → PgVectorRetriever (Module 4)
--
-- The filters mirror SearchFilters.matches() in backend/search/types.py,
-- so vector search (here) and BM25 (Python) agree on which jobs qualify.
-- Missing data is excluded when filtered on: a NULL salary can't prove ≥ X.
-- Remote: remote_type comes from the Enrichment Agent (Module 6); until
-- then the scrapers put "Remote" in location, so both are checked.
--
-- Note: the HNSW index finds nearest neighbours first, then the WHERE
-- clause filters them. With very selective filters a query can return
-- fewer than match_count rows — fine at our scale; pgvector ≥ 0.8 can fix
-- it with `SET hnsw.iterative_scan = relaxed_order`.
-- =============================================================
-- The signature changed (new filter args + more output columns), and
-- CREATE OR REPLACE can't change a return type — drop the old version first.
DROP FUNCTION IF EXISTS match_jobs(vector, double precision, integer);

CREATE OR REPLACE FUNCTION match_jobs(
    query_embedding     vector(384),
    match_threshold     float       DEFAULT 0.0,
    match_count         int         DEFAULT 20,
    filter_remote_only  boolean     DEFAULT false,
    filter_min_salary   int         DEFAULT NULL,
    filter_posted_after timestamptz DEFAULT NULL,
    filter_sources      text[]      DEFAULT NULL
)
RETURNS TABLE (
    id          UUID,
    title       TEXT,
    company     TEXT,
    location    TEXT,
    source      TEXT,
    url         TEXT,
    match_score FLOAT,
    similarity  FLOAT
)
LANGUAGE SQL STABLE
AS $$
    SELECT
        j.id, j.title, j.company, j.location, j.source, j.url, j.match_score,
        1 - (j.embedding <=> query_embedding) AS similarity   -- <=> is cosine DISTANCE
    FROM jobs j
    WHERE j.embedding IS NOT NULL
      AND 1 - (j.embedding <=> query_embedding) > match_threshold
      AND (NOT filter_remote_only
           OR j.remote_type = 'remote'
           OR j.location ILIKE '%remote%')
      AND (filter_min_salary IS NULL
           OR GREATEST(COALESCE(j.salary_max, 0), COALESCE(j.salary_min, 0)) >= filter_min_salary)
      AND (filter_posted_after IS NULL OR j.posted_at >= filter_posted_after)
      AND (filter_sources IS NULL OR j.source = ANY(filter_sources))
    ORDER BY j.embedding <=> query_embedding   -- nearest first; this is what uses the HNSW index
    LIMIT match_count;
$$;

-- =============================================================
-- Tell the Supabase API (PostgREST) to pick up the new tables now.
-- Without this you can get "Could not find the table 'public.jobs'
-- in the schema cache" (PGRST205) for a short while after creating them.
-- =============================================================
NOTIFY pgrst, 'reload schema';
