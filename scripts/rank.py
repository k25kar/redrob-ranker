#!/usr/bin/env python3
"""Ranking step (CPU-only, no network, deterministic).

Pipeline:
  load candidates -> rule score over all candidates -> drop impossible profiles
  -> bounded availability/location/verification adjustment
  -> near-tie refinement by role-relevant quality -> sort -> top 100 -> submission.csv

The order is the interpretable rule score (evidence of real work, seniority, experience, skills,
education) with small, capped adjustments. No external model, no network.

Run: python scripts/rank.py --candidates data/candidates.jsonl --config config.yaml
"""
import argparse, csv, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from redrob_ranker.conf import load_config, sha256_file
from redrob_ranker.scoring import rule_score, final_score
from redrob_ranker.gates import is_impossible
from redrob_ranker.quality import quality_score

# specific phrases to cite in reasoning (Stage-4: specific, varied, grounded — never invented)
STRONG_EVID = ["learning to rank", "dense retrieval", "vector search", "semantic search",
               "faiss", "pinecone", "hnsw", "bm25", "recsys", "recommender", "ndcg", "mrr",
               "rag", "a/b test", "embeddings"]


def evidence_phrases(c):
    p, hist = c["profile"], c["career_history"]
    blob = (p.get("summary", "") + " " + " ".join(h.get("description", "") for h in hist)).lower()
    return [e for e in STRONG_EVID if e in blob]


def build_reasoning_meta(title, yoe, resp, comps, reasons, phrases):
    """Cite the candidate's SPECIFIC matched evidence (varied, grounded — Stage-4 friendly)."""
    bits = [f"{title}, {yoe:.1f} yrs"]
    if phrases:
        bits.append("hands-on " + ", ".join(phrases[:3]))
    elif comps["evidence"] > 0.4:
        bits.append("ranking/recommendation experience in career history")
    elif comps["title"] >= 0.9:
        bits.append("strong AI/ML title, lighter explicit IR evidence")
    bits.append(f"recruiter response rate {resp:.2f}")
    s = "; ".join(bits) + "."
    if reasons:
        s += " Concern: " + ", ".join(reasons) + "."
    return s[:300]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = cfg["paths"]["outputs_dir"]
    os.makedirs(out, exist_ok=True)
    qt = cfg.get("quality_tiebreak", {"enabled": False})
    lam = qt.get("lambda", 0.0) if qt.get("enabled", False) else 0.0
    t0 = time.time()

    # integrity check (informational): warn if the dataset differs from the one we tuned on.
    try:
        if sha256_file(a.candidates) != cfg["integrity"]["expected_sha256"]:
            print("[warn] dataset hash differs from expected; ranking proceeds (rules work on any pool).")
    except Exception:
        pass

    # stream candidates: score ALL with the rule model (no recall risk) + hard gate. Keep compact records.
    rows = []     # (cid, final_score, comps, penalty, reasons)
    meta = {}     # cid -> (title, yoe, response_rate, evidence_phrases) for the top-100 writeout
    n = gated = 0
    for line in open(a.candidates):
        if not line.strip():
            continue
        c = json.loads(line); n += 1
        if is_impossible(c):
            gated += 1
            continue
        r = rule_score(c, cfg)
        fs = final_score(r["raw"], c["redrob_signals"], cfg, profile=c["profile"])
        cid = c["candidate_id"]
        p = c["profile"]
        # quality refinement: small weight, only reorders near-tied fit scores (see redrob_ranker/quality.py)
        q = quality_score(p, c["career_history"]) if qt.get("enabled", True) else 0.0
        rows.append((cid, fs + lam * q, r["comps"], r["penalty"], r["reasons"]))
        meta[cid] = (p.get("current_title", "?"), p.get("years_of_experience", 0),
                     c["redrob_signals"].get("recruiter_response_rate", 0.0), evidence_phrases(c))

    # final order: score desc, then candidate_id ascending (spec tie-break).
    # `s` already includes the small quality term; divide by (1+lam) so the displayed score stays in
    # [0,1] without changing the order or monotonicity.
    denom = 1.0 + lam
    final_rows = sorted(((cid, round(s / denom, 4), comps, pen, reasons)
                         for cid, s, comps, pen, reasons in rows),
                        key=lambda x: (-x[1], x[0]))
    top = final_rows[:100]

    sub_path = os.path.join(out, "submission.csv")
    with open(sub_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, s, comps, pen, reasons) in enumerate(top, 1):
            title, yoe, resp, phrases = meta[cid]
            w.writerow([cid, rank, f"{s:.4f}", build_reasoning_meta(title, yoe, resp, comps, reasons, phrases)])

    insp_path = os.path.join(out, "top50_inspect.csv")
    with open(insp_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "candidate_id", "title", "yoe", "score",
                    "evidence", "title_fit", "exp", "skills", "edu", "penalty", "reasons"])
        for rank, (cid, s, cm, pen, reasons) in enumerate(top[:50], 1):
            title, yoe, _, _ = meta[cid]
            w.writerow([rank, cid, title, f"{yoe:.1f}", f"{s:.4f}",
                        f"{cm['evidence']:.2f}", f"{cm['title']:.2f}", f"{cm['exp']:.2f}",
                        f"{cm['skills']:.2f}", f"{cm['edu']:.2f}", f"{pen:.2f}", "|".join(reasons)])

    print(f"read={n} gated(impossible)={gated} scored={len(rows)} emitted={len(top)} "
          f"elapsed={time.time()-t0:.1f}s")
    print(f"wrote {sub_path}, {insp_path}")


if __name__ == "__main__":
    main()
