# AI Job Search Aggregator

AI-powered job aggregator — pulls jobs from free job APIs (RemoteOK, The Muse,
Arbeitnow, Himalayas) every 6 hours, scores every job against your resume using
LangGraph agents, and surfaces the best matches in a Streamlit dashboard.

**Stack:** FastAPI · LangGraph · LangChain · Groq (free LLM) · Supabase ·
pgvector · Playwright · Apify · Sentence Transformers · Langfuse · APScheduler · Streamlit

---

## Architecture

How the main parts of the system connect. Solid arrows are the main data path;
dotted arrows are cross-cutting concerns (scheduling, tracing, evals).

```mermaid
flowchart LR
    subgraph Sources["Job sources (free)"]
        RO[RemoteOK API]
        TM[The Muse API]
        AN[Arbeitnow API]
        HI[Himalayas API]
        AP[Apify actors]
    end

    subgraph Ingest["Ingestion — Module 3"]
        SC[Scrapers<br/>→ list of RawJob]
        DD{Dedup<br/>SHA-256 hash}
    end

    subgraph DB["Supabase Postgres — Module 2"]
        JT[(jobs table<br/>+ pgvector HNSW)]
        AT[(applications)]
        SR[(scrape_runs)]
    end

    subgraph AI["AI pipeline — Modules 4, 6, 7"]
        EA[Enrichment Agent<br/>skills, seniority, remote, salary]
        SA[Scoring Agent<br/>job vs resume, 0–100]
        RT[Resume Tailor Agent<br/>bullets + ATS keywords]
        EM[Embeddings<br/>all-MiniLM-L6-v2, 384 dims]
        LLM[[Groq LLM<br/>llama-3.3-70b]]
    end

    subgraph Search["Hybrid search — Module 4"]
        BM[BM25 keyword]
        VS[pgvector similarity]
        RRF[RRF merge]
        RR[Cross-encoder rerank]
    end

    API[FastAPI — Module 8]
    UI[Streamlit dashboard<br/>top matches + Kanban — Module 12]
    MAIL[Email digest — Module 11]

    SCH([APScheduler<br/>every 6h — Module 11])
    LF([Langfuse tracing — Module 9])
    EV([RAGAS eval suite in CI — Modules 10, 14])

    RO & TM & AN & HI & AP --> SC --> DD -->|new only| JT
    SC -.-> SR
    JT -->|unenriched| EA --> SA --> RT --> JT
    EA & SA & RT <--> LLM
    JT -->|unembedded| EM --> JT
    API --> BM & VS
    BM & VS --> RRF --> RR --> API
    JT --- VS
    API --> UI
    UI <-->|save / apply| AT
    JT -->|score ≥ threshold| MAIL

    SCH -.-> SC
    SCH -.-> EA
    LLM -.-> LF
    EV -.->|gates prompt changes| AI
```

## Job data flow

What happens to a single job from the moment it's scraped until you act on it.

```mermaid
flowchart TD
    A[Scraper fetches listing] --> B[Build RawJob<br/>title, company, location, description, url, source]
    B --> C[Compute dedup_hash<br/>SHA-256 of title + company + location]
    C --> D{Hash already<br/>in jobs table?}
    D -->|yes| X[Skip — duplicate]
    D -->|no| E[Insert row<br/>enriched_at = NULL]

    E --> F[Enrichment Agent picks it up<br/>get_unenriched]
    F --> G[Groq extracts:<br/>required skills, tech stack,<br/>seniority, remote type, salary]
    G --> H[Scoring Agent compares to resume<br/>match_score, skills_match, skills_gap]
    H --> I[Resume Tailor Agent<br/>bullet suggestions, ATS keywords]
    I --> J[save_enrichment<br/>enriched_at = now, model + prompt_version logged]

    J --> K[Embedding pipeline<br/>get_unembedded]
    K --> L[save_embedding<br/>384-dim vector → HNSW index]

    L --> M{match_score ≥ 75?}
    M -->|yes| N[Included in email digest]
    M -->|no| O[Stays searchable only]

    L --> P[Hybrid search<br/>BM25 + vector → RRF → rerank]
    N --> Q
    P --> Q[Shown in Streamlit dashboard]

    Q --> R[You save it → applications.status = saved]
    R --> S[applied → screening → interview]
    S --> T{Outcome}
    T --> U[offer]
    T --> V[rejected / withdrawn]
```

---

## Quickstart (Windows — uv)

### 1. Install uv (once, global)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Restart your terminal after installing.

### 2. Clone / copy project
```powershell
cd C:\Projects
# if using git:
git clone https://github.com/sureshkunku/ai-job-search.git
cd ai-job-search

# or just copy the folder here
```

### 3. Create virtual environment + install deps (uv does both in one step)
```powershell
uv sync
```
This creates `.venv\` and installs all dependencies. Much faster than pip.

### 4. Install dev dependencies too
```powershell
uv sync --dev
```

### 5. Copy env file and fill in your keys
```powershell
copy .env.example .env
# open .env in PyCharm and fill in keys (see Free Keys section below)
```

### 6. Run tests (Modules 1–2 — no API keys needed, the DB is mocked)
```powershell
uv run pytest backend/tests -v
```

### 6b. Create the database tables (once)
Open your Supabase project → **SQL Editor**, paste in `scripts/setup_db.sql`, and run it.

### 7. Run the app (Module 5+)
```powershell
uv run uvicorn backend.main:app --reload
```

---

## Get Your Free API Keys

| Service | URL | What it's for | Free tier |
|---------|-----|----------------|-----------|
| Groq | https://console.groq.com | LLM (Llama 3.3 70B) — key starts with `gsk_` | Free tier |
| Supabase | https://supabase.com | Postgres + pgvector | 500MB free |
| Apify | https://apify.com | Extra job scraping | Free monthly credit |
| Langfuse | https://cloud.langfuse.com | LLM tracing | Free cloud |
| Resend or Gmail SMTP | https://resend.com | Email digest (Module 11) | Free tier |

RemoteOK, The Muse, Arbeitnow and Himalayas need **no API keys**.

---

## Project Structure

```
ai-job-search/
├── backend/
│   ├── config.py          # all settings (pydantic-settings)
│   ├── main.py            # FastAPI app
│   ├── models/            # Pydantic data models
│   ├── api/               # FastAPI route handlers
│   ├── scrapers/          # RemoteOK, The Muse, Arbeitnow, Himalayas, Apify
│   ├── agents/            # LangGraph AI agents
│   ├── db/                # Supabase repository layer
│   ├── scheduler/         # APScheduler jobs
│   ├── notifications/     # Email digest (Resend / Gmail SMTP)
│   └── tests/             # Pytest test suite
├── frontend/
│   └── app.py             # Streamlit dashboard
├── scripts/
│   └── setup_db.sql       # Supabase schema
├── data/
│   └── resume.txt         # YOUR resume (gitignored)
├── pyproject.toml
└── .env.example
```

---

## Course Modules

| Module | Topic | Status |
|--------|-------|--------|
| 1 | Project foundation, Pydantic models, Config | ✅ Done |
| 2 | Supabase schema, pgvector, Repository pattern | ✅ Done |
| 3 | Scrapers (RemoteOK, The Muse, Arbeitnow, Himalayas, Apify) | 🔄 Next |
| 4 | Embeddings, RAG, BM25, Hybrid search | ⏳ |
| 5 | Tool calling, MCP server | ⏳ |
| 6 | LangGraph agents (Enrichment, Scoring, Tailor) | ⏳ |
| 7 | Prompt engineering + versioning | ⏳ |
| 8 | FastAPI backend | ⏳ |
| 9 | Observability (Langfuse, MLOps) | ⏳ |
| 10 | Eval framework (RAGAS, golden dataset) | ⏳ |
| 11 | Scheduler + automation + email digest | ⏳ |
| 12 | Streamlit dashboard | ⏳ |
| 13 | Testing (pytest, mocking, eval suite) | ⏳ |
| 14 | CI/CD + Deploy | ⏳ |
