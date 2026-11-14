"""
chunking.py — split job descriptions and resumes into meaningful pieces.
Module 4: Embeddings + RAG

WHY chunk at all:
    An embedding model squeezes a text into ONE vector. all-MiniLM-L6-v2
    only reads the first ~256 word-pieces — anything after that is silently
    ignored. A 900-word job description would be judged on its intro alone
    ("We're a fast-growing startup..."), never reaching the requirements.

WHY chunk by SECTION instead of fixed-size token windows:
    Job descriptions and resumes are structured documents. "Requirements",
    "Nice to have" and "Benefits" mean very different things. Fixed windows
    cut across those boundaries, so one vector ends up half "Python, AWS"
    and half "dental insurance". Splitting at headings keeps each chunk about
    ONE thing, which is what makes chunk-level retrieval (e.g. "find the
    resume section that proves Kubernetes experience" in Module 6) precise.
    Only a section that is still too long gets a word-window split, with a
    small overlap so a sentence cut in half still has context on both sides.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Heading text → canonical section name. Matching is on the start of the line.
JOB_SECTIONS: dict[str, tuple[str, ...]] = {
    "about": ("about", "who we are", "the company", "our mission", "company description",
              "über uns", "wer wir sind"),
    "role": ("the role", "about the role", "role description", "job description", "overview",
             "position", "your role", "the opportunity", "deine rolle", "ihre aufgaben"),
    "responsibilities": ("responsibilities", "what you'll do", "what you will do",
                         "your responsibilities", "key responsibilities", "duties",
                         "day to day", "in this role", "deine aufgaben", "aufgaben"),
    "requirements": ("requirements", "qualifications", "what you'll bring", "what you bring",
                     "what we're looking for", "what we are looking for", "must have",
                     "you have", "who you are", "skills", "experience",
                     "dein profil", "ihr profil", "anforderungen"),
    "nice_to_have": ("nice to have", "nice-to-have", "bonus", "preferred", "plus",
                     "good to have", "wünschenswert"),
    "benefits": ("benefits", "perks", "what we offer", "why join", "compensation",
                 "salary", "we offer", "wir bieten", "das bieten wir"),
}

RESUME_SECTIONS: dict[str, tuple[str, ...]] = {
    "summary": ("summary", "profile", "professional summary", "objective", "about me"),
    "experience": ("experience", "work experience", "professional experience",
                   "employment", "work history", "career history"),
    "skills": ("skills", "technical skills", "core skills", "technologies", "tech stack",
               "core competencies", "tools"),
    "projects": ("projects", "key projects", "personal projects"),
    "education": ("education", "academic", "qualifications"),
    "certifications": ("certifications", "certificates", "licenses", "awards"),
}

_BULLET_PREFIX = re.compile(r"^[\s#*•\-–—>]+")


@dataclass(frozen=True)
class Chunk:
    section: str   # canonical section name, e.g. "requirements"
    text: str
    index: int     # position in the document (0 = first chunk)


def detect_heading(line: str, sections: dict[str, tuple[str, ...]]) -> str | None:
    """
    Return the canonical section if `line` looks like a heading, else None.

    A line is a heading when it is SHORT (≤ 7 words) and either:
      - IS a known phrase:                      "Requirements", "Nice to have"
      - starts with one AND is marked as a heading by a trailing colon or
        markdown:                               "What you'll do at Acme:", "## Skills & tools"
      - starts with a phrase that is only ever a heading: "About Acme", "Why join us"
    WHY so strict: "Experience with AWS" is a requirement, not a heading —
    treating it as one would split the requirements list in the wrong place.
    """
    stripped = line.strip()
    marked = stripped.endswith(":") or stripped.startswith(("#", "**", "__"))
    cleaned = _BULLET_PREFIX.sub("", stripped).strip().strip("*_").rstrip(":").strip().lower()
    if not cleaned or len(cleaned.split()) > 7:
        return None
    for name, phrases in sections.items():
        for phrase in phrases:
            if cleaned == phrase:
                return name
            if cleaned.startswith(phrase + " ") and (marked or phrase in _ALWAYS_HEADING_PREFIXES):
                return name
    return None


# Phrases that start a heading even without ":" or markdown ("About Acme").
_ALWAYS_HEADING_PREFIXES = {"about", "über uns", "why join", "what we offer", "wir bieten"}


def window_split(text: str, max_words: int, overlap: int) -> list[str]:
    """Split text into ≤ max_words windows, each repeating `overlap` words of the previous one."""
    words = text.split()
    if len(words) <= max_words:
        return [" ".join(words)] if words else []
    step = max(1, max_words - overlap)
    windows = []
    for start in range(0, len(words), step):
        windows.append(" ".join(words[start:start + max_words]))
        if start + max_words >= len(words):
            break
    return windows


def chunk_by_sections(
    text: str,
    sections: dict[str, tuple[str, ...]],
    *,
    default_section: str,
    max_words: int = 200,
    overlap: int = 30,
    start_index: int = 0,
) -> list[Chunk]:
    """Group lines under their nearest heading, then window-split long sections."""
    groups: list[tuple[str, list[str]]] = [(default_section, [])]
    for line in text.splitlines():
        heading = detect_heading(line, sections)
        if heading:
            groups.append((heading, []))
        elif line.strip():
            groups[-1][1].append(line.strip())

    chunks: list[Chunk] = []
    for section, lines in groups:
        for piece in window_split(" ".join(lines), max_words, overlap):
            chunks.append(Chunk(section=section, text=piece, index=start_index + len(chunks)))
    return chunks


def chunk_job(
    title: str,
    company: str,
    description: str,
    *,
    location: str | None = None,
    skills: list[str] | None = None,
    max_words: int = 200,
    overlap: int = 30,
) -> list[Chunk]:
    """
    Chunks for one job. Chunk 0 is always a compact "header" — title, company,
    location, skills — because those few words carry the most signal and must
    never be diluted by a long description.
    """
    header = f"{title} at {company}"
    if location:
        header += f" ({location})"
    if skills:
        header += ". Skills: " + ", ".join(skills)
    body = chunk_by_sections(description, JOB_SECTIONS, default_section="role",
                             max_words=max_words, overlap=overlap, start_index=1)
    return [Chunk(section="header", text=header, index=0), *body]


def chunk_resume(text: str, *, max_words: int = 200, overlap: int = 30) -> list[Chunk]:
    """Chunks for a resume, one section (or window of a long section) each."""
    return chunk_by_sections(text, RESUME_SECTIONS, default_section="summary",
                             max_words=max_words, overlap=overlap)
