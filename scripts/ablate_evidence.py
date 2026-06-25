#!/usr/bin/env python3
"""Buzzword-masking ablation (the decisive 'template vs real signal' test).

Mask retrieval buzzwords in candidate prose, recompute evidence + final scores, and measure how
much the top-20 changes. Also masks evidence ENTIRELY (weight->0) to see what structured signals
alone do. And exports rank-bucket samples for the manual hire-rate read.

Run: python scripts/ablate_evidence.py --candidates data/candidates.jsonl
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from redrob_ranker.conf import load_config
from redrob_ranker.scoring import final_score
from redrob_ranker.gates import is_impossible, soft_penalties
from redrob_ranker import features as F

# ChatGPT's exact list (+ close variants so masking is thorough)
MASK = ["learning-to-rank", "learning to rank", "ltr", "bm25", "faiss", "hnsw", "rag",
        "retrieval", "ndcg", "mrr", "vector search", "dense retrieval", "pinecone"]
_MASK_RE = re.compile("|".join(re.escape(m) for m in sorted(MASK, key=len, reverse=True)))


def masked_prose(p, hist):
    blob = (p.get("summary", "") + " " + " ".join(h.get("description", "") for h in hist)).lower()
    return _MASK_RE.sub(" [token] ", blob)


def evidence_on_text(blob, hist_descs):
    """Replicate features.evidence_score but on already-masked text (recency-weighted)."""
    import math
    score = 0.0
    for i, d in enumerate(hist_descs):
        w = math.exp(-0.25 * i)
        score += w * sum(1 for e in F.EVIDENCE if e in d)
    score += 0.5 * sum(1 for e in F.EVIDENCE if e in blob)   # summary approx via full blob
    return max(0.0, min(1.0, 1 - math.exp(-score / 3.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    w = cfg["weights"]

    recs = {}
    s_base, s_mask, s_noev = {}, {}, {}
    for line in open(a.candidates):
        if not line.strip():
            continue
        c = json.loads(line)
        if is_impossible(c):
            continue
        cid = c["candidate_id"]; recs[cid] = c
        p, hist = c["profile"], c["career_history"]
        # structured comps computed ONCE
        title = F.title_fit(p, hist); exp = F.experience_band(p.get("years_of_experience", 0))
        skills = F.skills_corroboration(c, p, hist); edu = F.education_score(c)
        ev_base = F.evidence_score(p, hist)
        descs = [_MASK_RE.sub(" [token] ", h.get("description", "").lower()) for h in hist]
        ev_mask = evidence_on_text(masked_prose(p, hist), descs)
        pen, _ = soft_penalties(c, cfg)
        struct = w["title"] * title + w["exp"] * exp + w["skills"] * skills + w["edu"] * edu

        def fin(ev):
            raw = max(0.0, struct + w["evidence"] * ev - pen)
            return final_score(raw, c["redrob_signals"], cfg)
        s_base[cid] = fin(ev_base); s_mask[cid] = fin(ev_mask); s_noev[cid] = fin(0.0)

    base = sorted(s_base, key=lambda k: (-s_base[k], k))
    masked = sorted(s_mask, key=lambda k: (-s_mask[k], k))
    noev = sorted(s_noev, key=lambda k: (-s_noev[k], k))

    def overlap(a_, b_, k):
        return len(set(a_[:k]) & set(b_[:k]))

    print("=== ABLATION: buzzword masking (bm25/faiss/hnsw/rag/retrieval/ndcg/LTR/...) ===")
    for k in (10, 20, 50, 100):
        print(f"  top-{k} overlap  baseline vs MASKED : {overlap(base, masked, k)}/{k}")
    print()
    for k in (10, 20, 50, 100):
        print(f"  top-{k} overlap  baseline vs NO-EVIDENCE (structured only): {overlap(base, noev, k)}/{k}")

    print("\n=== baseline top-10 vs masked top-10 (titles) ===")
    bt = {cid: recs[cid]["profile"]["current_title"] for cid in base[:10]}
    mt = {cid: recs[cid]["profile"]["current_title"] for cid in masked[:10]}
    print("baseline:", [bt[c] for c in base[:10]])
    print("masked  :", [mt[c] for c in masked[:10]])
    moved_out = [c for c in base[:20] if c not in set(masked[:20])]
    print(f"\nof baseline top-20, {len(moved_out)} dropped out of masked top-20")

    # rank-bucket samples for manual hire-rate read
    out = cfg["paths"]["outputs_dir"]
    with open(os.path.join(out, "rank_buckets.txt"), "w") as f:
        for lo, hi in ((1, 20), (80, 100), (300, 320)):
            f.write(f"\n{'#'*70}\nBUCKET ranks {lo}-{hi}\n{'#'*70}\n")
            for rk in range(lo, hi + 1):
                if rk - 1 >= len(base):
                    break
                cid = base[rk - 1]; c = recs[cid]; p = c["profile"]
                desc = " ".join(h.get("description", "") for h in c["career_history"])[:200]
                f.write(f"\n[{rk}] {cid} | {p['current_title']} | {p['years_of_experience']}y\n")
                f.write(f"     summary: {p.get('summary','')[:200]}\n")
                f.write(f"     work:    {desc}\n")
    print(f"\nwrote {out}/rank_buckets.txt (ranks 1-20, 80-100, 300-320 for manual read)")


if __name__ == "__main__":
    main()
