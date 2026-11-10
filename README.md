# AI Job Search Aggregator

AI-powered job aggregator — pulls jobs from free job APIs (RemoteOK, The Muse,
Arbeitnow, Himalayas) every 6 hours, scores every job against your resume using
LangGraph agents, and surfaces the best matches in a Streamlit dashboard.

**Stack:** FastAPI · LangGraph · LangChain · Groq (free LLM) · Supabase ·
pgvector · Playwright · Apify · Sentence Transformers · Langfuse · APScheduler · Streamlit

---

## Architecture

Four diagrams, from the big picture down. Each box is labelled with the module
that builds it — see [Course Modules](#course-modules) for status. Only
Modules 1–2 exist so far; everything else is the plan.

**Colour key (same in every diagram):**
🟦 Ingestion · 🟩 Database · 🟪 Embeddings + search · 🟧 Agents · 🩵 Tools + MCP ·
🩷 Prompts · 🟨 LLM · 🔵 API + UI · ⬛ Observability · 🟥 Evals · 🟫 Automation · 🔷 CI/CD.
Dashed boxes are planned tables that don't exist yet.

**Design choices (free-tools rule):** free job APIs instead of LinkedIn/Indeed
scraping · Resend or Gmail SMTP instead of SendGrid · a Postgres checkpointer
instead of Redis · salary tool uses stored job salaries instead of SerpAPI.

### 1. System architecture — every module

```mermaid
%%{init: {"flowchart": {"curve": "basis", "nodeSpacing": 30, "rankSpacing": 60, "padding": 14}, "themeVariables": {"fontFamily": "Segoe UI, Helvetica, Arial, sans-serif", "fontSize": "15px"}}}%%
flowchart TB
    subgraph M11["🟫 Module 11 · Automation"]
        direction LR
        SCH(["⏰ APScheduler<br/>every 6h"])
        CLEAN("🗑️ Stale-job cleanup<br/>after 30 days")
        MAIL("📧 Email digest<br/>Resend / Gmail")
        SCH ~~~ CLEAN ~~~ MAIL
    end

    subgraph M3["🟦 Module 3 · Ingestion"]
        direction LR
        SRC("🌐 Free job APIs<br/>RemoteOK · The Muse<br/>Arbeitnow · Himalayas · Apify") --> ORC("⚙️ Orchestrator<br/>asyncio fan-out") --> BS("🔁 BaseScraper<br/>retry · rate limit · backoff") --> NORM("🧹 Normalize<br/>→ RawJob") --> DD{"🔑 Dedup<br/>SHA-256"}
    end

    subgraph M2["🟩 Module 2 · Supabase Postgres + pgvector"]
        direction LR
        JT[("🗄️ jobs<br/>+ HNSW index")]
        AT[("📋 applications")]
        SR[("📝 scrape_runs")]
        PRT[("prompts<br/>planned")]
        FBT[("feedback<br/>planned")]
        ALT[("audit_log<br/>planned")]
        JT ~~~ AT ~~~ SR ~~~ PRT ~~~ FBT ~~~ ALT
    end

    subgraph M6["🟧 Module 6 · LangGraph agents"]
        direction LR
        SUP("🧠 Supervisor<br/>fan-out") --> SG{"🛡️ Safety<br/>gate"} --> EA("🏷️ Enrich") --> SA("📊 Score") --> RT("✍️ Tailor") --> HITL("🙋 Human<br/>review")
        CA("💬 Chat agent<br/>+ memory")
        CP[("💾 Checkpointer")]
        HITL ~~~ CA ~~~ CP
    end

    subgraph M4["🟪 Module 4 · Embeddings + hybrid search"]
        direction LR
        CH("🧩 Chunk<br/>by section") --> EMB("🔢 Embed<br/>MiniLM · 384d") --> RV("👤 Resume<br/>vector")
        BM("🔤 BM25<br/>keyword") --> RRF("🔀 RRF merge<br/>+ filters")
        VS("🧭 pgvector<br/>similarity") --> RRF --> RR("🏆 Rerank<br/>top 20 → 5")
    end

    subgraph BRAIN["🟨 Shared AI services"]
        direction LR
        LLM[["⚡ Groq<br/>llama-3.3-70b"]]
        REG("📚 Module 7 · Prompt registry<br/>versions · A/B · rollback")
    end

    subgraph SERVE["🔵 Serving · Modules 5, 8, 12"]
        direction LR
        EXT("🖱️ Claude Desktop<br/>/ MCP client") --> MCP("🔌 MCP server<br/>API-key auth") --> TOOLS("🧰 5 LangChain<br/>tools")
        MW("🔒 Auth +<br/>rate limit") --> API("🚀 FastAPI<br/>jobs · search · stats") --> UI("🖥️ Streamlit<br/>feed · Kanban · chat")
    end

    subgraph OPS["⬛ Quality + operations · Modules 9, 10, 14"]
        direction LR
        LF("🔭 Langfuse<br/>traces + cost")
        PROM("📈 Prometheus")
        CB("🧯 Circuit<br/>breaker")
        DRIFT("📉 Drift<br/>checks")
        GD("🥇 Golden<br/>dataset") --> RAGAS("📏 RAGAS")
        GD --> JUDGE("⚖️ LLM<br/>judge")
        GHA("🤖 GitHub<br/>Actions") --> DEP("🐳 Docker<br/>Railway")
        LF ~~~ PROM ~~~ CB ~~~ DRIFT ~~~ GD
        JUDGE ~~~ GHA
    end

    %% every arrow points downward so the layers stack top → bottom
    M11 -.->|"triggers every 6h"| M3
    M11 -.->|"reads for digest · archives old"| M2
    M3 ==>|"new jobs only"| M2
    M2 <==>|"unenriched jobs ⇄ scores + suggestions"| M6
    M2 <==>|"chunks ⇄ vectors"| M4
    M2 <-->|"applications · 👍👎"| SERVE
    M6 <-->|"LLM calls + prompts"| BRAIN
    M6 <-->|"chat"| SERVE
    M4 ==>|"ranked results"| SERVE
    BRAIN -.->|"traces · cost · eval gate"| OPS

    classDef ingest fill:#2563EB,stroke:#1E3A8A,color:#fff,stroke-width:2px
    classDef db fill:#16A34A,stroke:#14532D,color:#fff,stroke-width:2px
    classDef planned fill:#F0FDF4,stroke:#16A34A,color:#14532D,stroke-width:2px,stroke-dasharray:6 4
    classDef rag fill:#7C3AED,stroke:#4C1D95,color:#fff,stroke-width:2px
    classDef agent fill:#EA580C,stroke:#7C2D12,color:#fff,stroke-width:2px
    classDef tool fill:#0D9488,stroke:#134E4A,color:#fff,stroke-width:2px
    classDef prompt fill:#DB2777,stroke:#831843,color:#fff,stroke-width:2px
    classDef llm fill:#FACC15,stroke:#854D0E,color:#422006,stroke-width:3px
    classDef serve fill:#4F46E5,stroke:#312E81,color:#fff,stroke-width:2px
    classDef obs fill:#475569,stroke:#0F172A,color:#fff,stroke-width:2px
    classDef eval fill:#E11D48,stroke:#881337,color:#fff,stroke-width:2px
    classDef auto fill:#B45309,stroke:#78350F,color:#fff,stroke-width:2px
    classDef cicd fill:#0891B2,stroke:#164E63,color:#fff,stroke-width:2px
    classDef ext fill:#FFFFFF,stroke:#334155,color:#0F172A,stroke-width:2px

    class SRC,ORC,BS,NORM,DD ingest
    class JT,AT,SR db
    class PRT,FBT,ALT planned
    class CH,EMB,RV,BM,VS,RRF,RR rag
    class SUP,SG,EA,SA,RT,HITL,CA,CP agent
    class TOOLS,MCP tool
    class REG prompt
    class LLM llm
    class MW,API,UI serve
    class LF,PROM,CB,DRIFT obs
    class GD,RAGAS,JUDGE eval
    class SCH,MAIL,CLEAN auto
    class GHA,DEP cicd
    class EXT ext

    style M11 fill:#FFFBEB,stroke:#B45309,stroke-width:2px,color:#78350F
    style M3 fill:#EFF6FF,stroke:#2563EB,stroke-width:2px,color:#1E3A8A
    style M2 fill:#F0FDF4,stroke:#16A34A,stroke-width:2px,color:#14532D
    style M6 fill:#FFF7ED,stroke:#EA580C,stroke-width:2px,color:#7C2D12
    style M4 fill:#F5F3FF,stroke:#7C3AED,stroke-width:2px,color:#4C1D95
    style BRAIN fill:#FEFCE8,stroke:#CA8A04,stroke-width:2px,color:#713F12
    style SERVE fill:#EEF2FF,stroke:#4F46E5,stroke-width:2px,color:#312E81
    style OPS fill:#F8FAFC,stroke:#475569,stroke-width:2px,color:#0F172A
    linkStyle default stroke:#64748B,stroke-width:2.5px
```

### 2. Job data flow — one job's lifecycle

```mermaid
%%{init: {"flowchart": {"curve": "basis", "nodeSpacing": 40, "rankSpacing": 45}, "themeVariables": {"fontFamily": "Segoe UI, Helvetica, Arial, sans-serif", "fontSize": "15px"}}}%%
flowchart TD
    A(["⏰ Scheduler fires every 6h"]) --> B("⚙️ Orchestrator fans out to scrapers<br/>RemoteOK · The Muse · Arbeitnow · Himalayas · Apify")
    B --> C("🧹 Normalize any source → RawJob")
    C --> D("🔑 dedup_hash = SHA-256 of title + company + location")
    D --> E{"Hash already<br/>in jobs?"}
    E -->|yes| X("⏭️ Skip · counted as duplicate in scrape_runs")
    E -->|no| F[("🗄️ Insert row · enriched_at = NULL")]

    F --> G{"🛡️ Safety gate<br/>within token + cost budget?"}
    G -->|no| W("⏸️ Wait for next run")
    G -->|yes| H("🏷️ Enrichment Agent<br/>skills · seniority · remote · salary")
    H --> I("📊 Scoring Agent<br/>match_score 0–100 vs your resume")
    I --> J("✍️ Resume Tailor Agent<br/>RAG over resume chunks → bullets + ATS keywords")
    J --> K[("💾 save_enrichment<br/>+ LLM model · prompt_version · tokens · cost")]

    K --> L("🔢 Chunk JD by section → batch embed → save_embedding")
    L --> M{"match_score<br/>≥ 75?"}
    M -->|yes| N("📧 Email digest + high-confidence review")
    M -->|no| O("🔍 Searchable only")
    L --> P("🔀 Hybrid search<br/>BM25 + vector → RRF → rerank")
    N --> Q("🖥️ Seen in dashboard · chat · MCP clients")
    O --> Q
    P --> Q

    Q --> R("👍👎 Feedback → golden dataset")
    Q --> S("📋 saved → applied → screening → interview")
    S --> T{"Outcome"}
    T --> U(["🎉 offer"])
    T --> V(["rejected / withdrawn"])
    F -.->|"older than 30 days"| Z("🗑️ Archived by stale-job cleanup")

    classDef auto fill:#B45309,stroke:#78350F,color:#fff,stroke-width:2px
    classDef ingest fill:#2563EB,stroke:#1E3A8A,color:#fff,stroke-width:2px
    classDef db fill:#16A34A,stroke:#14532D,color:#fff,stroke-width:2px
    classDef agent fill:#EA580C,stroke:#7C2D12,color:#fff,stroke-width:2px
    classDef rag fill:#7C3AED,stroke:#4C1D95,color:#fff,stroke-width:2px
    classDef serve fill:#4F46E5,stroke:#312E81,color:#fff,stroke-width:2px
    classDef eval fill:#E11D48,stroke:#881337,color:#fff,stroke-width:2px
    classDef decide fill:#FEF3C7,stroke:#D97706,color:#78350F,stroke-width:2px
    classDef bad fill:#FEE2E2,stroke:#DC2626,color:#7F1D1D,stroke-width:2px
    classDef good fill:#22C55E,stroke:#14532D,color:#fff,stroke-width:3px
    classDef muted fill:#F1F5F9,stroke:#64748B,color:#334155,stroke-width:2px

    class A,N,Z auto
    class B,C,D ingest
    class F,K db
    class H,I,J agent
    class L,P rag
    class Q,S serve
    class R eval
    class E,G,M,T decide
    class X,W,V bad
    class U good
    class O muted
    linkStyle default stroke:#64748B,stroke-width:2px
```

### 3. Agents, tools and MCP — Modules 5, 6, 7

```mermaid
%%{init: {"flowchart": {"curve": "basis", "nodeSpacing": 35, "rankSpacing": 50}, "themeVariables": {"fontFamily": "Segoe UI, Helvetica, Arial, sans-serif", "fontSize": "15px"}}}%%
flowchart TD
    subgraph BATCH["🟧 Batch enrichment · runs on schedule"]
        Q[("🗄️ jobs where enriched_at IS NULL")] --> SUP("🧠 Supervisor<br/>routes jobs · parallel fan-out")
        SUP --> SG{"🛡️ Safety gate<br/>token budget · max iterations<br/>daily cost"}
        SG -->|over budget| STOP("⛔ Stop + log")
        SG -->|ok| EA("🏷️ Enrichment node<br/>→ EnrichmentOutput")
        EA --> SA("📊 Scoring node<br/>resume from state → ScoringOutput")
        SA --> RT("✍️ Resume Tailor node<br/>RAG over resume chunks<br/>→ ResumeTailorOutput")
        RT --> CONF{"High-confidence<br/>match?"}
        CONF -->|yes| HITL("🙋 Human-in-the-loop<br/>you approve or reject")
        CONF -->|no| SAVE[("💾 save_enrichment")]
        HITL --> SAVE
    end

    subgraph CHAT["🩵 Conversational search"]
        USER(["🧑 You: remote FastAPI roles over 150k"]) --> CA("💬 Conversation agent<br/>sliding window + summary memory")
        CA -->|LLM chooses a tool| TC{"🧰 Tool call<br/>Pydantic input schema"}
        TC --> T1("🔍 search_jobs")
        TC --> T2("📄 get_job_details")
        TC --> T3("📋 check_application_status")
        TC --> T4("🧭 search_similar_jobs")
        TC --> T5("💰 get_market_salary")
        T1 & T2 & T3 & T4 & T5 -->|result or handled error| CA
        CA --> ANS(["✅ Answer · can chain another tool"])
    end

    subgraph MCPBOX["🔌 MCP server"]
        CD("🖱️ Claude Desktop<br/>or any MCP client") -->|API key| MCP("🔌 Your MCP server")
        MCP --> T1 & T2 & T3 & T4 & T5
    end

    REG("📚 Prompt registry<br/>name + version + hash<br/>A/B split · rollback without deploy")
    LLM[["⚡ Groq LLM"]]
    CP[("💾 Checkpointer<br/>resume interrupted runs")]

    REG -.->|prompt| EA & SA & RT & CA
    EA & SA & RT & CA <--> LLM
    SUP -.- CP

    classDef agent fill:#EA580C,stroke:#7C2D12,color:#fff,stroke-width:2px
    classDef tool fill:#0D9488,stroke:#134E4A,color:#fff,stroke-width:2px
    classDef db fill:#16A34A,stroke:#14532D,color:#fff,stroke-width:2px
    classDef prompt fill:#DB2777,stroke:#831843,color:#fff,stroke-width:2px
    classDef llm fill:#FACC15,stroke:#854D0E,color:#422006,stroke-width:3px
    classDef decide fill:#FEF3C7,stroke:#D97706,color:#78350F,stroke-width:2px
    classDef bad fill:#FEE2E2,stroke:#DC2626,color:#7F1D1D,stroke-width:2px
    classDef good fill:#22C55E,stroke:#14532D,color:#fff,stroke-width:2px
    classDef ext fill:#FFFFFF,stroke:#334155,color:#0F172A,stroke-width:2px

    class SUP,EA,SA,RT,HITL,CA agent
    class T1,T2,T3,T4,T5,MCP tool
    class Q,SAVE,CP db
    class REG prompt
    class LLM llm
    class SG,CONF,TC decide
    class STOP bad
    class ANS good
    class USER,CD ext

    style BATCH fill:#FFF7ED,stroke:#EA580C,stroke-width:2px,color:#7C2D12
    style CHAT fill:#F0FDFA,stroke:#0D9488,stroke-width:2px,color:#134E4A
    style MCPBOX fill:#F0FDFA,stroke:#0D9488,stroke-width:2px,stroke-dasharray:6 4,color:#134E4A
    linkStyle default stroke:#64748B,stroke-width:2px
```

### 4. Evaluation, feedback and MLOps — Modules 9, 10, 14

```mermaid
%%{init: {"flowchart": {"curve": "basis", "nodeSpacing": 35, "rankSpacing": 55}, "themeVariables": {"fontFamily": "Segoe UI, Helvetica, Arial, sans-serif", "fontSize": "15px"}}}%%
flowchart LR
    subgraph PROD["⬛ Production monitoring · Module 9"]
        RUN("🤖 Every agent run") --> LF("🔭 Langfuse<br/>trace · tokens · latency · cost")
        RUN --> VER("🏷️ Logged per job<br/>LLM model · embedding model<br/>prompt_version")
        LF --> BUD{"Daily cost<br/>over budget?"}
        BUD -->|yes| ALERT("🚨 Alert")
        APIX("🚀 FastAPI") --> PROM("📈 Prometheus<br/>counters + latency histograms")
        APIX --> HC("❤️ Health check<br/>DB · LLM · scrapers")
        GROQ("⚡ Groq calls") --> CB{"🧯 Circuit breaker<br/>open?"}
        CB -->|yes| CACHE("💾 Serve cached scores")
        VER --> DRIFT("📉 Drift checks<br/>data: JDs changing<br/>concept: scores vs feedback")
    end

    subgraph LOOP["🟥 Feedback → eval loop · Module 10"]
        UI("🖥️ Dashboard 👍 / 👎") --> FB[("🗄️ feedback")]
        FB -->|disagreements| GD("🥇 Golden dataset<br/>50+ labelled jobs · versioned in git")
        GD --> EVAL("🧪 Eval runner")
        EVAL --> RAGAS("📏 RAGAS<br/>faithfulness · relevancy<br/>context precision + recall")
        EVAL --> JUDGE("⚖️ LLM-as-judge<br/>rubric 0–3 per dimension")
    end

    subgraph CICD["🔷 CI/CD · Module 14"]
        PRQ("🔀 Pull request<br/>new prompt or code") --> GHA("🤖 GitHub Actions<br/>ruff → mypy → pytest → evals")
        GHA --> EVAL
        RAGAS & JUDGE --> GATE{"Correctness down > 2%<br/>or cost up > 20%?"}
        GATE -->|yes| BLOCK("⛔ Block merge")
        GATE -->|no| BUILD("🐳 Docker build") --> DEP(["🚀 Deploy<br/>Railway + Streamlit Cloud"])
    end

    FB -.-> DRIFT
    DRIFT -.->|reprompt| PRQ

    classDef obs fill:#475569,stroke:#0F172A,color:#fff,stroke-width:2px
    classDef llm fill:#FACC15,stroke:#854D0E,color:#422006,stroke-width:2px
    classDef serve fill:#4F46E5,stroke:#312E81,color:#fff,stroke-width:2px
    classDef eval fill:#E11D48,stroke:#881337,color:#fff,stroke-width:2px
    classDef db fill:#16A34A,stroke:#14532D,color:#fff,stroke-width:2px
    classDef cicd fill:#0891B2,stroke:#164E63,color:#fff,stroke-width:2px
    classDef decide fill:#FEF3C7,stroke:#D97706,color:#78350F,stroke-width:2px
    classDef bad fill:#FEE2E2,stroke:#DC2626,color:#7F1D1D,stroke-width:2px
    classDef good fill:#22C55E,stroke:#14532D,color:#fff,stroke-width:3px

    class RUN,LF,VER,PROM,HC,DRIFT,CACHE obs
    class GROQ llm
    class APIX,UI serve
    class GD,EVAL,RAGAS,JUDGE eval
    class FB db
    class PRQ,GHA,BUILD cicd
    class BUD,CB,GATE decide
    class ALERT,BLOCK bad
    class DEP good

    style PROD fill:#F8FAFC,stroke:#475569,stroke-width:2px,color:#0F172A
    style LOOP fill:#FFF1F2,stroke:#E11D48,stroke-width:2px,color:#881337
    style CICD fill:#ECFEFF,stroke:#0891B2,stroke-width:2px,color:#164E63
    linkStyle default stroke:#64748B,stroke-width:2px
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
