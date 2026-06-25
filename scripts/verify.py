#!/usr/bin/env python3
"""Print verifiable artifacts: dataset hash/size, submission head/tail, consistency containment.
Run:  python scripts/verify.py --candidates data/candidates.jsonl --submission outputs/submission.csv
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from redrob_ranker.conf import sha256_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--submission", required=True)
    a = ap.parse_args()

    print("sha256:", sha256_file(a.candidates))
    n = sum(1 for _ in open(a.candidates)); print("lines :", n)
    print("bytes :", os.path.getsize(a.candidates))

    sub = open(a.submission).read().splitlines()
    print("\nhead -1:", sub[0]); print("row 1  :", sub[1]); print("tail -1:", sub[-1])
    print("rows   :", len(sub) - 1)

    # consistency containment: tenure-sum > YOE  -> must be 0 in top-100
    hp = []
    for line in open(a.candidates):
        c = json.loads(line); p = c["profile"]; hist = c["career_history"]
        if sum(h.get("duration_months", 0) for h in hist) > p["years_of_experience"] * 12 + 24:
            hp.append(c["candidate_id"])
    ids = set(l.split(",")[0] for l in sub[1:])
    print(f"\nimpossible-history profiles total: {len(hp)} | in submission top-100: {len(set(hp) & ids)}")


if __name__ == "__main__":
    main()
