"""Ranking metrics — reproduces the official composite exactly (for offline eval once labels exist).

    Composite = 0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10
Relevance in {0,1,2,3}; "relevant" for P@k/MAP means rel >= REL_THRESHOLD (tier 3+).
Label-driven => runs only with a gold/eval set, never inside rank.py.
"""
from __future__ import annotations
import math
from typing import Mapping, Sequence

REL_THRESHOLD = 3


def dcg(rels: Sequence[float], k: int) -> float:
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def ndcg(ranked_rels: Sequence[float], k: int) -> float:
    idcg = dcg(sorted(ranked_rels, reverse=True), k)
    return dcg(ranked_rels, k) / idcg if idcg > 0 else 0.0


def precision_at_k(ranked_rels, k, thr=REL_THRESHOLD):
    return sum(1 for r in ranked_rels[:k] if r >= thr) / k if k else 0.0


def average_precision(ranked_rels, thr=REL_THRESHOLD):
    hits, score = 0, 0.0
    for i, r in enumerate(ranked_rels):
        if r >= thr:
            hits += 1
            score += hits / (i + 1)
    n = sum(1 for r in ranked_rels if r >= thr)
    return score / n if n else 0.0


def composite(ranked_ids: Sequence[str], rel: Mapping[str, float]) -> dict:
    rr = [rel.get(cid, 0) for cid in ranked_ids]
    out = {"NDCG@10": ndcg(rr, 10), "NDCG@50": ndcg(rr, 50),
           "MAP": average_precision(rr), "P@10": precision_at_k(rr, 10)}
    out["composite"] = 0.50 * out["NDCG@10"] + 0.30 * out["NDCG@50"] + 0.15 * out["MAP"] + 0.05 * out["P@10"]
    return out
