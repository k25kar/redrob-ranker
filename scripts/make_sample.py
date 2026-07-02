#!/usr/bin/env python3
"""Make a small candidates sample for the sandbox demo.

Accepts either the bundle's sample_candidates.json (pretty-printed JSON array) or the full
candidates.jsonl, and writes a JSONL sample the ranker can consume directly.

  python scripts/make_sample.py --input sample_candidates.json --out data/sample.jsonl
  python scripts/make_sample.py --input candidates.jsonl --n 100 --out data/sample.jsonl
"""
import argparse, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help=".json array or .jsonl")
    ap.add_argument("--out", default="data/sample.jsonl")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()

    with open(a.input) as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":                      # pretty JSON array (sample_candidates.json)
            cands = json.load(f)[: a.n]
        else:                                # jsonl
            cands = []
            for line in f:
                if line.strip():
                    cands.append(json.loads(line))
                if len(cands) >= a.n:
                    break

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for c in cands:
            f.write(json.dumps(c) + "\n")
    print(f"wrote {len(cands)} candidates -> {a.out}")


if __name__ == "__main__":
    main()
