"""
test_search.py — Module 4: embeddings, chunking, BM25, RRF, rerank, hybrid search.

Fast by default: a FakeEmbedder (hashed bag-of-words) replaces the real
model, so no 90 MB download and no torch start-up. It deliberately maps
"ml" / "machine learning" to the same dimension — a tiny stand-in for what
real embeddings do — so we can show dense search catching a paraphrase
that keyword search misses.

Real-model tests (download ~180 MB once) run only with:
    RUN_SLOW_TESTS=1 uv run pytest backend/tests/test_search.py -m slow
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from backend.db.jobs_repo import JobsRepo
from backend.search import (
    BM25Index,
    EmbeddingPipeline,
    HybridSearcher,
    InMemoryDenseRetriever,
    PgVectorRetriever,
    ResumeIndex,
    SearchDoc,
    SearchFilters,
    chunk_job,
    chunk_resume,
    embed_job_documents,
    reciprocal_rank_fusion,
    tokenize,
)
from backend.search.chunking import JOB_SECTIONS, detect_heading, window_split
from backend.search.embeddings import mean_pool, normalize

# ===========================================================================
# Fakes
# ===========================================================================

_SYNONYMS = {"ml": "machinelearning", "machine": "machinelearning", "learning": "machinelearning",
             "ai": "artificialintelligence", "llm": "artificialintelligence"}


class FakeEmbedder:
    """Deterministic bag-of-words embedder: each (normalized) word → one hashed dimension."""

    def __init__(self, dimensions: int = 384) -> None:
        self._dims = dimensions
        self.calls = 0
        self.texts_embedded = 0

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        self.texts_embedded += len(texts)
        out = np.zeros((len(texts), self._dims), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                token = _SYNONYMS.get(token, token)
                out[row, int(hashlib.md5(token.encode()).hexdigest(), 16) % self._dims] += 1.0
            if not out[row].any():
                out[row, 0] = 1.0
        return normalize(out)


class FakeReranker:
    """Scores by how many query words appear in the job TITLE (titles matter most)."""

    def __init__(self) -> None:
        self.calls = 0

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls += 1
        q = set(tokenize(query))
        return [float(len(q & set(tokenize(t.split(" at ")[0])))) for t in texts]


def doc(id_: str, title: str, description: str = "", **kw) -> SearchDoc:
    return SearchDoc(id=id_, title=title, company=kw.pop("company", "Co"),
                     description=description, **kw)


NOW = datetime.now(UTC)
DOCS = [
    doc("py", "Senior Python Engineer", "Build FastAPI services on AWS.",
        location="Remote", salary_max=170000, posted_at=NOW - timedelta(days=2), source="remoteok"),
    doc("ml", "Machine Learning Engineer", "Train and deploy models with PyTorch.",
        location="Berlin", salary_max=120000, posted_at=NOW - timedelta(days=20), source="arbeitnow"),
    doc("fe", "Frontend Developer", "React and TypeScript design systems.",
        location="Austin TX", posted_at=NOW - timedelta(days=1), source="himalayas"),
    doc("de", "Data Engineer", "Spark, Databricks and Python pipelines.",
        location="Remote (worldwide)", salary_min=150000, posted_at=NOW - timedelta(days=5),
        source="himalayas"),
]


# ===========================================================================
# Chunking
# ===========================================================================

JD = """We are Acme, building AI for logistics.
Responsibilities:
- Build LangGraph agents
- Own the FastAPI backend
Requirements
- 5+ years Python
- Experience with AWS
Nice to have:
- Rust
Benefits
Remote-first, 30 days holiday"""


class TestChunking:
    def test_job_chunks_follow_sections(self):
        chunks = chunk_job("AI Engineer", "Acme", JD, location="Remote", skills=["Python"])
        assert [c.section for c in chunks] == [
            "header", "role", "responsibilities", "requirements", "nice_to_have", "benefits"]
        assert chunks[0].text == "AI Engineer at Acme (Remote). Skills: Python"
        assert "5+ years Python" in chunks[3].text and "Experience with AWS" in chunks[3].text
        assert [c.index for c in chunks] == list(range(6))

    @pytest.mark.parametrize(("line", "expected"), [
        ("Requirements", "requirements"),
        ("## What you'll do", "responsibilities"),
        ("**Nice to have**", "nice_to_have"),
        ("What you'll bring to Acme:", "requirements"),
        ("About Acme", "about"),
        ("Experience with AWS", None),                 # a requirement, not a heading
        ("Skills in Python and SQL are needed for this", None),
        ("", None),
    ])
    def test_detect_heading(self, line, expected):
        assert detect_heading(line, JOB_SECTIONS) == expected

    def test_long_section_is_window_split_with_overlap(self):
        words = [f"w{i}" for i in range(450)]
        windows = window_split(" ".join(words), max_words=200, overlap=30)
        assert len(windows) == 3
        assert all(len(w.split()) <= 200 for w in windows)
        assert windows[1].split()[0] == "w170"         # starts 30 words before window 0 ended

    def test_resume_sections(self):
        resume = "Jane Doe, Senior Engineer\nExperience\nBuilt RAG at X\nSkills:\nPython, AWS\nEducation\nBSc CS"
        assert [c.section for c in chunk_resume(resume)] == ["summary", "experience", "skills", "education"]


# ===========================================================================
# Embeddings helpers + pipeline
# ===========================================================================

class TestEmbeddings:
    def test_mean_pool_is_unit_length(self):
        pooled = mean_pool(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
        assert np.isclose(np.linalg.norm(pooled), 1.0)
        assert np.allclose(pooled, [2 ** -0.5, 2 ** -0.5, 0])

    def test_embed_job_documents_batches_all_chunks_in_one_call(self):
        embedder = FakeEmbedder()
        jobs = [{"title": "AI Engineer", "company": "A", "description": JD},
                {"title": "Data Engineer", "company": "B", "description": "Spark"}]
        vectors = embed_job_documents(embedder, jobs)
        assert vectors.shape == (2, 384)
        assert embedder.calls == 1                  # ONE batched model call
        assert embedder.texts_embedded == 6 + 2     # 6 chunks + 2 chunks
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


class TestEmbeddingPipeline:
    async def test_embeds_and_saves_every_unembedded_job(self):
        repo = MagicMock(spec=JobsRepo)
        repo.get_unembedded = AsyncMock(return_value=[
            {"id": "1", "title": "AI Engineer", "company": "A", "description": JD},
            {"id": "2", "title": "Data Engineer", "company": "B", "description": "Spark"},
            {"id": "3", "title": "ML Engineer", "company": "C", "description": None},
        ])
        repo.save_embedding = AsyncMock(side_effect=[None, RuntimeError("db"), None])

        report = await EmbeddingPipeline(repo, FakeEmbedder(), batch_size=2).run()

        assert (report.embedded, report.failed) == (2, 1)
        saved_ids = [c.args[0] for c in repo.save_embedding.call_args_list]
        assert saved_ids == ["1", "2", "3"]
        assert len(repo.save_embedding.call_args_list[0].args[1]) == 384

    async def test_refuses_wrong_dimensions(self):
        repo = MagicMock(spec=JobsRepo)
        with pytest.raises(ValueError, match="vector\\(384\\)"):
            await EmbeddingPipeline(repo, FakeEmbedder(dimensions=768)).run()


# ===========================================================================
# BM25, filters, dense, RRF
# ===========================================================================

class TestBM25:
    def test_tokenize_keeps_tech_terms(self):
        assert tokenize("C++ and C# with Node.js, the FastAPI way") == [
            "c++", "c#", "node.js", "fastapi", "way"]

    def test_exact_keyword_ranks_first(self):
        hits = BM25Index(DOCS).search("FastAPI python", k=5)
        assert hits[0][0] == "py"
        assert "fe" not in [h[0] for h in hits]     # shares no query word → excluded

    def test_filters_and_empty_cases(self):
        index = BM25Index(DOCS)
        assert [h[0] for h in index.search("python", filters=SearchFilters(sources=["himalayas"]))] == ["de"]
        assert index.search("the and of") == []      # only stopwords
        assert BM25Index([]).search("python") == []


class TestFilters:
    @pytest.mark.parametrize(("filters", "expected"), [
        (SearchFilters(), {"py", "ml", "fe", "de"}),
        (SearchFilters(remote_only=True), {"py", "de"}),             # via location text
        (SearchFilters(min_salary=150000), {"py", "de"}),            # max OR min ≥ 150k
        (SearchFilters(posted_within_days=7), {"py", "fe", "de"}),
        (SearchFilters(sources=["arbeitnow"]), {"ml"}),
        (SearchFilters(remote_only=True, min_salary=160000), {"py"}),
    ])
    def test_matches(self, filters, expected):
        assert {d.id for d in DOCS if filters.matches(d)} == expected

    def test_missing_salary_is_excluded(self):
        assert not SearchFilters(min_salary=1).matches(DOCS[2])


class TestDense:
    async def test_in_memory_cosine_and_filters(self):
        embedder = FakeEmbedder()
        vectors = embed_job_documents(embedder, [d.model_dump() for d in DOCS])
        retriever = InMemoryDenseRetriever(DOCS, vectors)
        query = embedder.embed(["ML engineer"])[0]   # "ML" ≈ "Machine Learning" in the fake

        hits = await retriever.search(query, k=2)
        assert hits[0][0] == "ml" and -1.0 <= hits[0][1] <= 1.0
        hits = await retriever.search(query, k=4, filters=SearchFilters(remote_only=True))
        assert {h[0] for h in hits} == {"py", "de"}

    async def test_pgvector_retriever_passes_filters_to_sql(self):
        repo = MagicMock(spec=JobsRepo)
        repo.semantic_search = AsyncMock(return_value=[
            {"id": "a", "similarity": 0.9, "match_score": 80},
            {"id": "b", "similarity": 0.8, "match_score": 40},
        ])
        f = SearchFilters(remote_only=True, min_salary=100000, posted_within_days=7,
                          sources=["remoteok"], min_match_score=50)
        hits = await PgVectorRetriever(repo).search(np.ones(4), k=10, filters=f)

        kwargs = repo.semantic_search.call_args.kwargs
        assert kwargs["remote_only"] is True and kwargs["min_salary"] == 100000
        assert kwargs["sources"] == ["remoteok"] and kwargs["limit"] == 10
        assert kwargs["posted_after"] < datetime.now(UTC)
        assert hits == [("a", 0.9)]                  # match_score filter applied in Python


class TestRRF:
    def test_exact_scores(self):
        fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60))
        assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
        assert fused["a"] == pytest.approx(1 / 61)

    def test_agreement_beats_a_single_first_place(self):
        fused = reciprocal_rank_fusion([["solo", "x", "both"], ["y", "both", "z"]], k=60)
        assert fused[0][0] == "both"                 # #3 + #2 beats one #1

    def test_empty(self):
        assert reciprocal_rank_fusion([[], []]) == []


# ===========================================================================
# Hybrid search end-to-end (fakes)
# ===========================================================================

class TestHybrid:
    async def test_hybrid_combines_both_retrievers(self):
        searcher = HybridSearcher.in_memory(DOCS, FakeEmbedder())
        response = await searcher.search("ML python", top_k=4)

        by_id = {h.doc.id: h for h in response.hits}
        # BM25 can't see "ML" in "Machine Learning Engineer"; dense can.
        assert by_id["ml"].dense_rank is not None and by_id["ml"].bm25_rank is None
        # "python" is a literal keyword → BM25 finds the Python/data jobs.
        assert by_id["py"].bm25_rank is not None
        assert [h.rank for h in response.hits] == list(range(1, len(response.hits) + 1))
        assert response.reranked is False and response.latency_ms >= 0

    async def test_reranker_sets_final_order(self):
        reranker = FakeReranker()
        searcher = HybridSearcher.in_memory(DOCS, FakeEmbedder(), reranker)
        response = await searcher.search("data engineer spark", top_k=2)
        assert response.reranked and reranker.calls == 1
        assert response.hits[0].doc.id == "de"
        assert response.hits[0].rerank_score >= response.hits[1].rerank_score

    async def test_filters_apply_to_both_retrievers(self):
        searcher = HybridSearcher.in_memory(DOCS, FakeEmbedder())
        response = await searcher.search("engineer python", SearchFilters(remote_only=True), top_k=10)
        assert {h.doc.id for h in response.hits} <= {"py", "de"}

    async def test_same_job_in_many_countries_shows_once(self):
        docs = [doc(f"m{i}", "Machine Learning Engineer", "PyTorch", company="mercor",
                    location=f"{country} (Remote)")
                for i, country in enumerate(["UK", "US", "India"])]
        docs.append(doc("other", "ML Platform Engineer", "PyTorch", company="Acme"))
        response = await HybridSearcher.in_memory(docs, FakeEmbedder()).search("machine learning pytorch")
        companies = [h.doc.company for h in response.hits]
        assert companies.count("mercor") == 1 and "Acme" in companies

    async def test_search_by_vector_with_resume(self):
        embedder = FakeEmbedder()
        searcher = HybridSearcher.in_memory(DOCS, embedder)
        resume = ResumeIndex.build("Summary\nFrontend developer, React and TypeScript", embedder)
        response = await searcher.search_by_vector(resume.vector, top_k=1)
        assert response.hits[0].doc.id == "fe"

    async def test_from_database_uses_pgvector(self):
        repo = MagicMock(spec=JobsRepo)
        repo.get_search_corpus = AsyncMock(return_value=[
            {"id": 1, "title": "AI Engineer", "company": "A", "description": None,
             "required_skills": None, "location": "Remote", "embedding": "ignored"},
        ])
        repo.semantic_search = AsyncMock(return_value=[
            {"id": "1", "similarity": 0.7},
            {"id": "999", "similarity": 0.6},        # not in corpus (added later) → skipped
        ])
        searcher = await HybridSearcher.from_database(repo, FakeEmbedder())
        response = await searcher.search("AI engineer")
        assert [h.doc.id for h in response.hits] == ["1"]
        assert response.hits[0].dense_score == 0.7


# ===========================================================================
# Resume index
# ===========================================================================

RESUME = """Jane Doe — Senior Platform Engineer
Experience
Built LangGraph agents and RAG pipelines on AWS.
Skills
Python, FastAPI, Kubernetes, Terraform
Education
BSc Computer Science"""


class TestResumeIndex:
    def test_cache_is_reused_and_invalidated(self, tmp_path):
        cache = tmp_path / "resume_index.json"
        embedder = FakeEmbedder()

        first = ResumeIndex.load_or_build(RESUME, embedder, model_name="m", cache_path=cache)
        assert cache.exists() and embedder.calls == 1
        assert json.loads(cache.read_text())["key"] == ResumeIndex.cache_key(RESUME, "m")

        again = ResumeIndex.load_or_build(RESUME, embedder, model_name="m", cache_path=cache)
        assert embedder.calls == 1                   # served from cache
        assert np.allclose(again.vector, first.vector, atol=1e-5)

        ResumeIndex.load_or_build(RESUME + "\nGo", embedder, model_name="m", cache_path=cache)
        assert embedder.calls == 2                   # resume changed → rebuilt

    def test_top_chunks_finds_relevant_section(self):
        embedder = FakeEmbedder()
        index = ResumeIndex.build(RESUME, embedder)
        (best, score), *_ = index.top_chunks(embedder.embed(["kubernetes terraform"])[0], k=2)
        assert best.section == "skills" and score > 0

    def test_empty_resume_rejected(self):
        with pytest.raises(ValueError):
            ResumeIndex.build("   ", FakeEmbedder())


# ===========================================================================
# Repository changes
# ===========================================================================

def _mock_client():
    client, qb = MagicMock(), MagicMock()
    qb.execute = AsyncMock(return_value=MagicMock(data=[]))
    for m in ("select", "eq", "is_", "order", "limit", "rpc"):
        getattr(qb, m).return_value = qb
    qb.not_ = MagicMock()
    client.table.return_value = qb
    client.rpc.return_value = qb
    return client, qb


class TestRepoChanges:
    async def test_get_unembedded_no_longer_waits_for_enrichment(self):
        client, qb = _mock_client()
        repo = JobsRepo()
        repo._supabase = client
        await repo.get_unembedded()
        qb.is_.assert_called_once_with("embedding", "null")
        qb.not_.is_.assert_not_called()

    async def test_semantic_search_sends_filters(self):
        client, _ = _mock_client()
        repo = JobsRepo()
        repo._supabase = client
        after = datetime(2026, 11, 1, tzinfo=UTC)
        await repo.semantic_search([0.1] * 384, limit=5, remote_only=True, posted_after=after,
                                   sources=["remoteok"])
        name, params = client.rpc.call_args.args
        assert name == "match_jobs"
        assert params["filter_remote_only"] is True and params["match_count"] == 5
        assert params["filter_posted_after"] == after.isoformat()
        assert params["filter_sources"] == ["remoteok"] and params["filter_min_salary"] is None

    async def test_search_corpus_excludes_vectors(self):
        client, qb = _mock_client()
        repo = JobsRepo()
        repo._supabase = client
        await repo.get_search_corpus()
        assert "embedding" not in qb.select.call_args.args[0]


# ===========================================================================
# Real models (slow, opt-in)
# ===========================================================================

slow = pytest.mark.skipif(not os.getenv("RUN_SLOW_TESTS"), reason="set RUN_SLOW_TESTS=1 to run")


@pytest.mark.slow
@slow
class TestRealModels:
    def test_real_embedder_understands_paraphrase(self):
        from backend.config import get_settings
        from backend.search import get_embedder

        s = get_settings()
        embedder = get_embedder(s.embedding_model)
        assert embedder.dimensions == s.embedding_dimensions
        v = embedder.embed(["machine learning engineer", "ML developer building models",
                            "pastry chef at a bakery"])
        assert v[0] @ v[1] > v[0] @ v[2] + 0.2

    def test_real_cross_encoder_prefers_relevant_job(self):
        from backend.config import get_settings
        from backend.search import CrossEncoderReranker

        reranker = CrossEncoderReranker(get_settings().reranker_model)
        relevant, irrelevant = reranker.score(
            "remote python backend engineer",
            ["Senior Python backend engineer, fully remote, FastAPI and AWS.",
             "Store manager for a retail clothing shop in Austin."])
        assert relevant > irrelevant
