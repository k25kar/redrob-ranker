#!/usr/bin/env python3
"""Ranking step (CPU-only, no network, deterministic). Rungs 1-2: rule score + BM25 recall.

Pipeline:
  load candidates -> rule_score(all 100K) -> hard gate (exclude impossible) -> behavioral blend
  -> BM25 recall pool (rung 2) -> [cross-encoder rerank = rung 3, TODO] -> submission.csv

BM25 currently feeds the rerank POOL + a recall diagnostic; it does not change the final order
until the cross-encoder (rung 3) is wired in. This is the ablation discipline: add, measure, keep.

Run: python scripts/rank.py --candidates data/candidates.jsonl --config config.yaml
"""
import argparse, csv, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from redrob_ranker.conf import load_config, sha256_file
from redrob_ranker.scoring import rule_score, final_score
from redrob_ranker.gates import is_impossible
from redrob_ranker.retrieval import BM25, rrf_fuse
from redrob_ranker.schema import candidate_text

# role/query terms for BM25 (derived from the JD)
JD_QUERY = ("senior ai engineer embeddings retrieval vector search ranking recommendation "
            "learning to rank nlp information retrieval production python product company "
            "hybrid search evaluation ndcg mrr ab testing recsys semantic search")

# Specific phrases to cite in reasoning (Stage-4: specific, varied, non-templated, grounded).
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
    out = cfg["paths"]["outputs_dir"]; adir = cfg["paths"]["artifacts_dir"]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    # --- integrity check: same pool -> use committed artifacts; mismatch -> fallback ---
    fallback = False
    bm25_path = os.path.join(adir, "bm25.pkl")
    try:
        digest = sha256_file(a.candidates)
        if digest != cfg["integrity"]["expected_sha256"]:
            fallback = True
            print(f"[warn] dataset hash mismatch -> FALLBACK (build BM25 at runtime, dense disabled)")
    except Exception:
        fallback = True

    # --- STREAM candidates: rule score over ALL (no recall risk) + hard gate.
    #     Keep only COMPACT records (no full dicts) to stay within the memory budget. ---
    rows = []                 # (id, final_score, comps, penalty, reasons)
    meta = {}                 # id -> small fields needed for reasoning/inspection of the top-100
    runtime_docs = None       # only populated in fallback (build BM25 at runtime)
    if fallback:
        runtime_docs = ([], [])
    n = gated = 0
    for line in open(a.candidates):
        if not line.strip():
            continue
        c = json.loads(line); n += 1
        cid = c["candidate_id"]
        if fallback:
            runtime_docs[0].append(cid); runtime_docs[1].append(candidate_text(c))
        if is_impossible(c):
            gated += 1
            continue
        r = rule_score(c, cfg)
        fs = final_score(r["raw"], c["redrob_signals"], cfg)
        rows.append((cid, fs, r["comps"], r["penalty"], r["reasons"]))
        p = c["profile"]
        meta[cid] = (p.get("current_title", "?"), p.get("years_of_experience", 0),
                     c["redrob_signals"].get("recruiter_response_rate", 0.0), evidence_phrases(c))

    # --- BM25 recall pool (rung 2): load committed index, else build at runtime (fallback) ---
    if not fallback and os.path.exists(bm25_path):
        bm = BM25.load(bm25_path)
    else:
        bm = BM25().fit(runtime_docs[0], runtime_docs[1])
    bm25_ranked = bm.search(JD_QUERY, top_k=cfg["retrieval"]["bm25_topN"])

    # compact tuple layout: (cid, fs, comps, penalty, reasons)
    rows.sort(key=lambda x: -x[1])
    rule_rank_of = {t[0]: i + 1 for i, t in enumerate(rows)}
    rule_ranked = rows[: cfg["retrieval"]["rule_topN"]]
    rule_list = [(t[0], t[1]) for t in rule_ranked]
    rerank_pool = rrf_fuse([rule_list, bm25_ranked], k=cfg["retrieval"]["rrf_k"],
                           top_k=cfg["retrieval"]["rerank_pool"])
    pool_ids = set(cid for cid, _ in rerank_pool)

    # recall diagnostic: BM25-surfaced candidates NOT in the rule top-N (does BM25 add anything?)
    rule_topN_ids = set(cid for cid, _ in rule_list)
    bm25_only = [(cid, s) for cid, s in bm25_ranked if cid not in rule_topN_ids][:40]

    # --- final ranking (rung 1-2: rule score; rung 3 cross-encoder will rerank pool_ids) ---
    final_rows = sorted(((cid, round(fs, 4), comps, pen, reasons)
                         for cid, fs, comps, pen, reasons in rows),
                        key=lambda x: (-x[1], x[0]))   # score desc, candidate_id asc
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
                    "evidence", "title_fit", "exp", "skills", "edu", "penalty", "in_bm25_pool", "reasons"])
        for rank, (cid, s, cm, pen, reasons) in enumerate(top[:50], 1):
            title, yoe, _, _ = meta[cid]
            w.writerow([rank, cid, title, f"{yoe:.1f}", f"{s:.4f}",
                        f"{cm['evidence']:.2f}", f"{cm['title']:.2f}", f"{cm['exp']:.2f}",
                        f"{cm['skills']:.2f}", f"{cm['edu']:.2f}", f"{pen:.2f}",
                        "Y" if cid in pool_ids else "N", "|".join(reasons)])

    diag_path = os.path.join(out, "retrieval_diag.csv")
    with open(diag_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bm25_rank", "candidate_id", "title", "yoe", "bm25_score", "rule_final_rank"])
        for i, (cid, s) in enumerate(bm25_only, 1):
            title, yoe, _, _ = meta.get(cid, ("(gated/absent)", 0, 0, []))
            w.writerow([i, cid, title, f"{yoe:.1f}", f"{s:.2f}", rule_rank_of.get(cid, ">topN")])

    print(f"read={n} gated(impossible)={gated} scored={len(rows)} "
          f"emitted={len(top)} fallback={fallback} elapsed={time.time()-t0:.1f}s")
    print(f"rerank pool={len(pool_ids)} | bm25-only-not-in-rule-topN (diag)={len(bm25_only)}")
    print(f"wrote {sub_path}, {insp_path}, {diag_path}")


if __name__ == "__main__":
    main()
