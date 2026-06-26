"""Rule score (embedding-free) plus the bounded availability/location adjustment.

The rule score is a weighted sum of interpretable components (evidence, title, experience, skills,
education) with soft penalties. The final score applies a small, capped availability and
location/verification adjustment. Both keep their natural spread, so the ranking is explainable.
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


def final_score(rule_raw: float, rs: dict, cfg: dict, profile: dict = None) -> float:
    """Bounded availability + location/notice fit on the RAW (spread-preserving) rule score.

    score = rule_raw * availability_multiplier * context_multiplier
    Both multipliers are capped near 1.0, so they reorder near-ties and ease down clear mismatches
    without burying a strong technical fit. profile is optional for backward compatibility.
    """
    b = cfg["behavioral"]
    s = rule_raw * B.availability_multiplier(rs, b["floor"], b["reach_threshold"])
    if profile is not None:
        s *= B.context_multiplier(profile, rs, cfg)
    return s
