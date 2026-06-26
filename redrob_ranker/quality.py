"""Within-cluster quality refinement.

Among candidates whose fit scores are nearly tied (an interchangeable strong cluster), a recruiter
still separates them. This module scores the signals the JD explicitly prioritises:
  - evaluation rigor (experience measuring ranking quality and running A/B tests) -- JD: make-or-break
  - seniority (Staff / Principal / Lead / Senior in the title)
  - ownership (owned / designed / led / built from scratch)
  - measurable impact (scale figures and metric improvements in the prose)

The result is in [0, 1] and is added to the fit score with a small weight (config: quality_tiebreak.
lambda), so it only reorders genuine near-ties and never lifts a weak-fit candidate across tiers.
It uses the same career-history text the fit score reads; it adds no external data.
"""
from __future__ import annotations
import re

SENIORITY = {"staff": 4, "principal": 4, "lead": 3, "senior": 2}
EVAL_TERMS = ["ndcg", "mrr", " map", "offline-online", "offline/online", "a/b test", "ab test",
              "evaluation harness", "eval framework", "recall@", "precision@"]
OWNERSHIP = ["owned", "designed", "led ", "built", "from scratch", "drove", "architected", "rebuilt"]
_SCALE = re.compile(r"\b\d+\s?(?:m|k|gb|million|billion)\b|\b\d+m\+|\bqps\b|\bp95\b|\d+ms\b")
_GAIN = re.compile(r"\d\.\d+\s*(?:to|->|→)\s*\d\.\d+|\b\d{1,3}%")


def _seniority(title: str) -> int:
    t = title.lower()
    for k, v in SENIORITY.items():
        if k in t:
            return v
    return 1


def quality_score(profile: dict, hist: list) -> float:
    blob = (profile.get("summary", "") + " " +
            " ".join(h.get("description", "") for h in hist)).lower()
    eval_rigor = min(6, sum(1 for e in EVAL_TERMS if e in blob)) / 6
    ownership = min(6, sum(1 for o in OWNERSHIP if o in blob)) / 6
    impact = min(8, len(_SCALE.findall(blob)) + len(_GAIN.findall(blob))) / 8
    seniority = (_seniority(profile.get("current_title", "")) - 1) / 3
    return 0.35 * eval_rigor + 0.25 * seniority + 0.25 * impact + 0.15 * ownership
