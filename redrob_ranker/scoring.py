"""Rule score (RAW, embedding-free) + behavioral blend + final-combine helpers.

IMPORTANT (reviewer fix): the rules-only score is the RAW weighted composite, which has natural
separation. We do NOT percentile-normalize it for the rules-only ranking — percentile-normalization
is reserved for the rung-3 fusion of rule⊕cross-encoder (different scales), per FINAL_IMPLEMENTATION_PLAN §0.
"""
from __future__ import annotations
from . import features as F
from . import gates as G
from . import behavioral as B


def rule_score(c: dict, cfg: dict) -> dict:
    """Return {raw, comps, penalty, reasons} — all embedding-free, fully explainable."""
    p, hist = c["profile"], c["career_history"]
    comps = {
        "evidence": F.evidence_score(p, hist),
        "title": F.title_fit(p, hist),
        "exp": F.experience_band(p.get("years_of_experience", 0),
                                 cfg["role_profile"]["experience_years"]["min"],
                                 cfg["role_profile"]["experience_years"]["max"]),
        "skills": F.skills_corroboration(c, p, hist),
        "edu": F.education_score(c),
    }
    w = cfg["weights"]
    raw = sum(w[k] * v for k, v in comps.items())
    pen, reasons = G.soft_penalties(c, cfg)
    return {"raw": max(0.0, raw - pen), "comps": comps, "penalty": pen, "reasons": reasons}


def final_score(rule_raw: float, rs: dict, cfg: dict) -> float:
    """Graded availability penalty on the RAW (spread-preserving) rule score.

    score = rule_raw * availability_multiplier   (multiplier in [floor, 1.0]).
    One clean behavioral mechanism — no additive blend, no binary cliff.
    """
    b = cfg["behavioral"]
    return rule_raw * B.availability_multiplier(rs, b["floor"], b["reach_threshold"])


def percentile_norm(scores: dict) -> dict:
    """Map id->score to id->percentile in [0,1] (used ONLY for rung-3 rule⊕CE fusion)."""
    order = sorted(scores, key=lambda k: scores[k])
    m = len(order)
    return {cid: (i / (m - 1) if m > 1 else 1.0) for i, cid in enumerate(order)}
