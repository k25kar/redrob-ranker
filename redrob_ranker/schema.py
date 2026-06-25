"""Typed-ish access to the candidate schema + safe handling of sentinels/missing fields.

Empirically (measured on the released 100K):
    github_activity_score == -1  -> 64.6% of candidates  (means "no GitHub linked", NOT "worst")
    offer_acceptance_rate == -1  -> 59.6%                 (means "no offer history",  NOT "worst")
Treating -1 as a low score would wreck ~60% of the pool. We map -1 -> None (missing) here.
"""
from __future__ import annotations
import json
from datetime import date
from typing import Iterator, Optional

TODAY = date(2026, 6, 25)


def iter_candidates(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_date(s) -> Optional[date]:
    try:
        return date.fromisoformat(s)
    except Exception:
        return None


def github_score(rs: dict) -> Optional[float]:
    v = rs.get("github_activity_score", -1)
    return None if v is None or v == -1 else float(v)  # -1 => missing


def offer_acceptance(rs: dict) -> Optional[float]:
    v = rs.get("offer_acceptance_rate", -1)
    return None if v is None or v == -1 else float(v)  # -1 => missing


def prose(p: dict, hist: list) -> str:
    """Searchable lowercase prose: summary + all role descriptions."""
    return (p.get("summary", "") + " " +
            " ".join(h.get("description", "") for h in hist)).lower()


def candidate_text(c: dict) -> str:
    """Retrieval text for BM25 — KEY FIELDS ONLY (headline, summary, titles, companies, skills).

    We deliberately exclude full role descriptions: they ~5x the index size/memory for little
    recall gain (titles+summary+skills already carry the lexical signal). Keeps BM25 lean enough
    to build/load well within the memory budget.
    """
    p, hist = c["profile"], c["career_history"]
    parts = [p.get("headline", ""), p.get("summary", ""), p.get("current_title", "")]
    for h in hist:
        parts.append(f"{h.get('title','')} {h.get('company','')}")
    parts += [s.get("name", "") for s in c.get("skills", [])]
    return " ".join(parts).lower()
