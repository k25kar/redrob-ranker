#!/usr/bin/env python3
"""Audit the evidence extractor + dump debug vectors. Answers the reviewer's concerns 1, 2, 6, 7.
Run: python scripts/audit.py --candidates data/candidates.jsonl
"""
import argparse, json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from redrob_ranker.conf import load_config
from redrob_ranker.scoring import rule_score, final_score
from redrob_ranker.gates import is_impossible
from redrob_ranker.behavioral import availability_norm, availability_multiplier
from redrob_ranker import features as F

NONTECH = ("hr manager", "accountant", "sales", "marketing", "graphic", "content writer",
           "customer support", "operations", "civil engineer", "mechanical engineer", "project manager")
STRONG_EVID = ("learning to rank", "ltr", "faiss", "dense retrieval", "vector search",
               "semantic search", "hnsw", "recsys", "recommender", "ndcg", "mrr", "bm25", "pinecone")
WEAK_EVID = ("ranking", "search", "matching", "personalization", "ctr", "recommendation")


def fired_phrases(c):
    p, hist = c["profile"], c["career_history"]
    blob = (p.get("summary", "") + " " + " ".join(h.get("description", "") for h in hist)).lower()
    return [e for e in F.EVIDENCE if e in blob]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)

    recs = {}
    evid_all = []
    nontech_hi = []          # non-tech title but evidence>=0.9  -> permissiveness signal
    for line in open(a.candidates):
        if not line.strip():
            continue
        c = json.loads(line)
        if is_impossible(c):
            continue
        r = rule_score(c, cfg)
        fs = final_score(r["raw"], c["redrob_signals"], cfg)
        p = c["profile"]; cm = r["comps"]
        ev = cm["evidence"]
        evid_all.append(ev)
        title = p.get("current_title", "").lower()
        if ev >= 0.9 and any(nt in title for nt in NONTECH):
            nontech_hi.append((c["candidate_id"], p.get("current_title"), ev, fired_phrases(c)[:8]))
        recs[c["candidate_id"]] = dict(c=c, r=r, fs=fs, comps=cm, raw=r["raw"], title=p.get("current_title"),
                                       yoe=p.get("years_of_experience", 0))

    # --- concern 1: evidence distribution + permissiveness ---
    import statistics
    print("=== EVIDENCE DISTRIBUTION (all scored) ===")
    for thr in (0.99, 0.9, 0.7, 0.5, 0.3, 0.1):
        print(f"  evidence >= {thr}: {sum(1 for x in evid_all if x>=thr)} ({100*sum(1 for x in evid_all if x>=thr)/len(evid_all):.1f}%)")
    print(f"  mean={statistics.mean(evid_all):.3f} median={statistics.median(evid_all):.3f}")
    print(f"\n=== PERMISSIVENESS TEST: NON-TECH titles with evidence>=0.9 ===")
    print(f"  count: {len(nontech_hi)}  (if high -> extractor too permissive)")
    for cid, t, ev, ph in nontech_hi[:10]:
        print(f"   {cid} | {t} | ev={ev:.2f} | fired={ph}")

    # --- concern 2: does evidence subsume skills? correlation in the top-500 ---
    order = sorted(recs, key=lambda k: -recs[k]["fs"])
    top500 = order[:500]
    import math
    ev = [recs[c]["comps"]["evidence"] for c in top500]
    sk = [recs[c]["comps"]["skills"] for c in top500]
    me, ms = sum(ev)/len(ev), sum(sk)/len(sk)
    cov = sum((e-me)*(s-ms) for e, s in zip(ev, sk))/len(ev)
    sde = math.sqrt(sum((e-me)**2 for e in ev)/len(ev)); sds = math.sqrt(sum((s-ms)**2 for s in sk)/len(sk))
    corr = cov/(sde*sds) if sde*sds else 0
    print(f"\n=== concern 2: corr(evidence, skills) in top-500 = {corr:.3f}  (high => double-counting) ===")

    # --- concern 6: debug vectors at requested ranks ---
    rank_of = {c: i+1 for i, c in enumerate(order)}
    print("\n=== DEBUG VECTORS (reconcile final = raw * avail_mult) ===")
    print(f"{'rank':>5} {'cid':>14} {'raw':>6} {'availM':>7} {'final':>7} | evid title exp skills edu pen | title")
    for rk in (1, 20, 100, 192, 500, 1000):
        if rk-1 >= len(order):
            continue
        cid = order[rk-1]; R = recs[cid]; cm = R["comps"]; rs = R["c"]["redrob_signals"]
        m = availability_multiplier(rs, cfg["behavioral"]["floor"], cfg["behavioral"]["reach_threshold"])
        print(f"{rk:>5} {cid:>14} {R['raw']:>6.3f} {m:>7.3f} {R['fs']:>7.3f} | "
              f"{cm['evidence']:.2f} {cm['title']:.2f} {cm['exp']:.2f} {cm['skills']:.2f} {cm['edu']:.2f} {R['r']['penalty']:.2f} | {R['title']}")

    # --- concern 7: export top-20 FULL profiles for the human read + strong/weak evidence tag ---
    with open(os.path.join(cfg["paths"]["outputs_dir"], "top20_profiles.txt"), "w") as f:
        for rk, cid in enumerate(order[:20], 1):
            c = recs[cid]["c"]; p = c["profile"]; cm = recs[cid]["comps"]
            ph = fired_phrases(c)
            strong = [x for x in ph if x in STRONG_EVID]; weak = [x for x in ph if x in WEAK_EVID]
            f.write(f"\n{'='*90}\nRANK {rk}  {cid}  | {p['current_title']} | {p['years_of_experience']}y | "
                    f"final={recs[cid]['fs']:.3f} evid={cm['evidence']:.2f} skills={cm['skills']:.2f}\n")
            f.write(f"STRONG-evidence phrases: {strong}\nWEAK/generic phrases: {weak}\n")
            f.write(f"SUMMARY: {p.get('summary','')}\n")
            for h in c['career_history'][:3]:
                f.write(f"  - {h['title']} @ {h['company']} ({h.get('duration_months',0)}mo): {h.get('description','')[:280]}\n")
    print(f"\nwrote {cfg['paths']['outputs_dir']}/top20_profiles.txt (full profiles + strong/weak evidence tags)")

    # quick strong-vs-weak tally for the top-50
    print("\n=== TOP-50: do they have STRONG evidence (real IR terms) or only WEAK/generic? ===")
    s_only = w_only = both = none = 0
    for cid in order[:50]:
        ph = fired_phrases(recs[cid]["c"])
        hs = any(x in STRONG_EVID for x in ph); hw = any(x in WEAK_EVID for x in ph)
        both += hs and hw; s_only += hs and not hw; w_only += hw and not hs; none += not hs and not hw
    print(f"  strong+weak={both}  strong-only={s_only}  weak/generic-only={w_only}  none={none}")


if __name__ == "__main__":
    main()
