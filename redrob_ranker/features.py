"""Embedding-free features (the decisive, explainable signals).

All return values are in [0,1]. NO embedding/semantic terms here — semantic matching is
confined to retrieval and the cross-encoder (FINAL_IMPLEMENTATION_PLAN.md §0).
"""
from __future__ import annotations
import math
from .schema import prose

# Title -> fit weight for the Senior AI Engineer (search/ranking/recsys) role.
AI_TITLES = {
    "ml engineer": 1.0, "machine learning engineer": 1.0, "applied ml engineer": 1.0,
    "ai engineer": 1.0, "senior ai engineer": 1.0, "lead ai engineer": 1.0,
    "staff machine learning engineer": 1.0, "senior machine learning engineer": 1.0,
    "recommendation systems engineer": 1.0, "search engineer": 1.0, "search relevance engineer": 1.0,
    "applied scientist": 0.95, "senior applied scientist": 0.95, "nlp engineer": 0.95,
    "senior nlp engineer": 0.95, "ai research engineer": 0.85, "research engineer": 0.8,
    "mlops engineer": 0.8, "data scientist": 0.78, "senior data scientist": 0.8,
    "ai specialist": 0.7, "software engineer": 0.55, "senior software engineer": 0.6,
    "backend engineer": 0.5, "data engineer": 0.55, "analytics engineer": 0.45,
    "full stack developer": 0.35, "computer vision engineer": 0.5,
}
JUNIOR = ("junior", "intern", "trainee", "associate", "fresher", "graduate engineer")
ML_TOKENS = ("ml", "machine learning", "ai", "deep learning", "nlp", "(ml)", "(ai)")

EVIDENCE = [
    "recommendation", "recommender", "ranking", "retrieval", "embedding", "vector search",
    "semantic search", "search relevance", "learning to rank", "ltr", "recsys",
    "information retrieval", "personalization", "candidate ranking", "bm25", "faiss",
    "pinecone", "elasticsearch", "opensearch", "qdrant", "weaviate", "ndcg", "mrr",
    "a/b test", "ab test", "click-through", "ctr", "matching",
]
PROD_EVIDENCE = ["production", "deployed", "real users", "at scale", "latency", "serving", "throughput"]
MUST_HAVE_SKILLS = [
    "embeddings", "retrieval", "vector", "rag", "semantic search", "sentence-transformers",
    "faiss", "pinecone", "elasticsearch", "bge", "e5", "learning to rank", "ranking",
    "recommendation", "nlp", "information retrieval", "python", "pytorch", "tensorflow",
]


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def title_fit(p: dict, hist: list) -> float:
    """Best fit over current + 3 most-recent titles. ML/AI tokens rescue under-mapped titles
    (e.g. 'Senior Software Engineer (ML)'); 'Junior/Intern/...' titles are down-weighted."""
    titles = [p.get("current_title", "")] + [h.get("title", "") for h in hist[:3]]
    best = 0.0
    for t in titles:
        tl = t.lower().strip()
        base = AI_TITLES.get(tl)
        if base is None:
            base = max((w for k, w in AI_TITLES.items() if k in tl), default=0.0)
        # ML/AI token bonus for titles our map underrates (caps at 0.8 via tokens alone)
        if any(tok in tl for tok in ML_TOKENS):
            base = max(base, 0.8)
        if any(j in tl for j in JUNIOR):     # seniority discount on the title itself
            base *= 0.5
        best = max(best, base)
    return _clip(best)


def is_junior_title(p: dict) -> bool:
    return any(j in p.get("current_title", "").lower() for j in JUNIOR)


def evidence_score(p: dict, hist: list) -> float:
    """Recency-weighted IR/ranking/recsys evidence in prose; product roles weighted higher."""
    score = 0.0
    for i, h in enumerate(hist):
        w = math.exp(-0.25 * i)                      # most-recent role weighted most
        d = h.get("description", "").lower()
        hits = sum(1 for e in EVIDENCE if e in d)
        prod = 1.3 if h.get("industry", "") != "IT Services" else 1.0
        score += w * hits * prod
    score += 0.5 * sum(1 for e in EVIDENCE if e in p.get("summary", "").lower())
    return _clip(1 - math.exp(-score / 3.0))         # ~6 weighted hits -> ~1.0


def experience_band(yoe: float, lo: int = 5, hi: int = 9) -> float:
    if yoe < lo:
        return _clip(yoe / lo) * 0.9
    if yoe <= hi:
        return 1.0
    return _clip(1 - 0.05 * (yoe - hi), 0.6, 1.0)


def skills_corroboration(c: dict, p: dict, hist: list) -> float:
    """Skills CORROBORATE (capped contribution); discounted by stuffer inconsistency."""
    skills = c.get("skills", [])
    if not skills:
        return 0.4
    names = [s["name"].lower() for s in skills]
    matched = sum(1 for m in MUST_HAVE_SKILLS if any(m in n or n in m for n in names))
    base = _clip(matched / 6.0)
    pr = prose(p, hist)
    incons = sum(1 for s in skills if s.get("endorsements", 0) == 0 and s["name"].lower() not in pr)
    return base * _clip(1 - 0.04 * incons, 0.6, 1.0)


def education_score(c: dict) -> float:
    tiers = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.6, "tier_4": 0.45, "unknown": 0.5}
    edu = c.get("education", [])
    return max((tiers.get(e.get("tier", "unknown"), 0.5) for e in edu), default=0.5)
