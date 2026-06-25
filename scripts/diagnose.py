#!/usr/bin/env python3
"""Diagnostics — measure, don't design. Answers:
  1) BM25 rescue value: for k in {100,200,400,800}, how many BM25 top-k are buried by rules
     (rule_rank>2000) AND look genuinely strong?  (anecdote -> statistic)
  2) Feature dominance: component vectors at ranks 1,5,10,25,50,100.
  3) Buried-strong cases: bm25_rank<=20 AND rule_rank>500 — list + deep-dive.
  4) Export rules_top200.csv + bm25_rescued.csv for manual reading.
Run: python scripts/diagnose.py --candidates data/candidates.jsonl
"""
import argparse, csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from redrob_ranker.conf import load_config
from redrob_ranker.scoring import rule_score, final_score
from redrob_ranker.gates import is_impossible
from redrob_ranker.retrieval import BM25
from redrob_ranker.schema import prose

JD_QUERY = ("senior ai engineer embeddings retrieval vector search ranking recommendation "
            "learning to rank nlp information retrieval production python product company "
            "hybrid search evaluation ndcg mrr ab testing recsys semantic search")
STRONG_TITLES = ("ml engineer", "machine learning", "ai engineer", "nlp", "applied scientist",
                 "recommendation", "search", "research engineer", "data scientist", "ai specialist")


def looks_strong(rec):
    t = rec["title"].lower()
    return (any(s in t for s in STRONG_TITLES) and rec["evidence"] >= 0.40
            and 5 <= rec["yoe"] <= 12 and rec["penalty"] < 0.10 and "junior" not in t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)

    recs = {}   # cid -> dict(title,yoe,evidence,title_fit,exp,skills,edu,penalty,final,reasons)
    for line in open(a.candidates):
        if not line.strip():
            continue
        c = json.loads(line)
        if is_impossible(c):
            continue
        r = rule_score(c, cfg)
        fs = final_score(r["raw"], c["redrob_signals"], cfg)
        p = c["profile"]; cm = r["comps"]
        recs[c["candidate_id"]] = dict(
            title=p.get("current_title", "?"), yoe=p.get("years_of_experience", 0),
            evidence=cm["evidence"], title_fit=cm["title"], exp=cm["exp"],
            skills=cm["skills"], edu=cm["edu"], penalty=r["penalty"], final=fs,
            reasons="|".join(r["reasons"]))

    order = sorted(recs, key=lambda cid: -recs[cid]["final"])
    rule_rank = {cid: i + 1 for i, cid in enumerate(order)}

    bm = BM25.load(os.path.join(cfg["paths"]["artifacts_dir"], "bm25.pkl"))
    bm25_ranked = bm.search(JD_QUERY, top_k=1000)
    bm25_rank = {cid: i + 1 for i, (cid, _s) in enumerate(bm25_ranked)}

    print("=== (1) BM25 RESCUE VALUE: BM25 top-k that rules bury (rule_rank>2000) ===")
    print(f"{'k':>5} {'buried(rule>2000)':>18} {'of-those-look-strong':>22}")
    for k in (100, 200, 400, 800):
        topk = [cid for cid, _ in bm25_ranked[:k]]
        buried = [cid for cid in topk if cid in rule_rank and rule_rank[cid] > 2000]
        strong = [cid for cid in buried if looks_strong(recs[cid])]
        # also count BM25 top-k that were gated as inconsistent (absent from recs)
        gated = sum(1 for cid in topk if cid not in recs)
        print(f"{k:>5} {len(buried):>18} {len(strong):>22}   (gated-inconsistent in top-{k}: {gated})")

    print("\n=== (2) FEATURE DOMINANCE: component vectors at key ranks ===")
    print(f"{'rank':>5} {'final':>7} {'evid':>5} {'title':>6} {'exp':>5} {'skills':>6} {'edu':>5} {'pen':>5}  title")
    for rk in (1, 5, 10, 25, 50, 100):
        cid = order[rk - 1]; r = recs[cid]
        print(f"{rk:>5} {r['final']:>7.4f} {r['evidence']:>5.2f} {r['title_fit']:>6.2f} "
              f"{r['exp']:>5.2f} {r['skills']:>6.2f} {r['edu']:>5.2f} {r['penalty']:>5.2f}  {r['title']}")

    print("\n=== (3) BURIED-STRONG: bm25_rank<=20 AND rule_rank>500 ===")
    cases = []
    for cid, _ in bm25_ranked[:20]:
        rr = rule_rank.get(cid)
        if cid in recs and rr and rr > 500:
            cases.append((cid, bm25_rank[cid], rr, recs[cid]))
    print(f"found {len(cases)} such cases in BM25 top-20")
    for cid, br, rr, r in cases:
        print(f"  {cid} bm25#{br} rule#{rr} | {r['title']} {r['yoe']}y | "
              f"evid={r['evidence']:.2f} title={r['title_fit']:.2f} exp={r['exp']:.2f} "
              f"skills={r['skills']:.2f} pen={r['penalty']:.2f} | {r['reasons']}")

    # broaden: bm25<=50 & rule>500 to estimate scale
    wide = [(cid, bm25_rank[cid], rule_rank[cid]) for cid, _ in bm25_ranked[:50]
            if cid in recs and rule_rank[cid] > 500]
    print(f"\n(bm25<=50 & rule>500: {len(wide)} cases)")

    # exports for manual reading
    out = cfg["paths"]["outputs_dir"]
    with open(os.path.join(out, "rules_top200.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["rule_rank", "candidate_id", "title", "yoe", "final",
                                       "evidence", "title_fit", "exp", "skills", "edu", "penalty", "reasons"])
        for i, cid in enumerate(order[:200], 1):
            r = recs[cid]; w.writerow([i, cid, r["title"], f"{r['yoe']:.1f}", f"{r['final']:.4f}",
                f"{r['evidence']:.2f}", f"{r['title_fit']:.2f}", f"{r['exp']:.2f}",
                f"{r['skills']:.2f}", f"{r['edu']:.2f}", f"{r['penalty']:.2f}", r["reasons"]])
    with open(os.path.join(out, "bm25_rescued.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["bm25_rank", "candidate_id", "title", "yoe", "rule_rank",
                                       "evidence", "title_fit", "exp", "skills", "penalty"])
        for cid, _ in bm25_ranked[:100]:
            if cid in recs and rule_rank[cid] > 200:
                r = recs[cid]; w.writerow([bm25_rank[cid], cid, r["title"], f"{r['yoe']:.1f}", rule_rank[cid],
                    f"{r['evidence']:.2f}", f"{r['title_fit']:.2f}", f"{r['exp']:.2f}",
                    f"{r['skills']:.2f}", f"{r['penalty']:.2f}"])
    print(f"\nwrote {out}/rules_top200.csv and {out}/bm25_rescued.csv")


if __name__ == "__main__":
    main()
