#!/usr/bin/env python3
"""OFFLINE precompute: build the BM25 index once and store artifacts/bm25.pkl + dataset hash.
(Dense embeddings would also be produced here on a free GPU; rung 5, not yet enabled.)
Run:  python scripts/build_artifacts.py --candidates data/candidates.jsonl
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from redrob_ranker.conf import load_config, sha256_file
from redrob_ranker.retrieval import BM25
from redrob_ranker.schema import candidate_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    adir = cfg["paths"]["artifacts_dir"]; os.makedirs(adir, exist_ok=True)

    ids, docs = [], []
    for line in open(a.candidates):
        if line.strip():
            c = json.loads(line); ids.append(c["candidate_id"]); docs.append(candidate_text(c))

    t = time.time()
    bm = BM25().fit(ids, docs)
    bm.save(os.path.join(adir, "bm25.pkl"))
    with open(os.path.join(adir, "dataset.sha256"), "w") as f:
        f.write(sha256_file(a.candidates))
    print(f"BM25 built over {len(ids)} docs in {time.time()-t:.1f}s -> {adir}/bm25.pkl")
    print("dataset.sha256 written.")


if __name__ == "__main__":
    main()
