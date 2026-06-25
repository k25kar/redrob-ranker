"""Rung 2 — BM25 retrieval (dependency-free, inverted-index for speed).

Role in the architecture: BM25 does NOT replace the rule score. It broadens the candidate
pool that gets the expensive cross-encoder (rung 3), so a strong candidate the rule score
under-rates on structured signals can still be reranked. Build once -> bm25.pkl (committed);
fall back to runtime build only on a dataset hash mismatch.
"""
from __future__ import annotations
import math
import pickle
import re
from collections import Counter, defaultdict

_TOK = re.compile(r"[a-z0-9+#.]+")
STOP = set("the a an and or of to in for with on at by is are be as i we our".split())


def tokenize(text: str) -> list:
    return [t for t in _TOK.findall(text.lower()) if t not in STOP and len(t) > 1]


class BM25:
    """Okabi BM25 with an inverted index; scores only docs containing query terms."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: list = []
        self.doc_len: list = []
        self.avgdl: float = 0.0
        self.df: dict = {}
        self.postings: dict = defaultdict(list)   # term -> [(doc_idx, tf), ...]
        self.N: int = 0

    def fit(self, ids: list, docs: list) -> "BM25":
        self.ids = ids
        self.N = len(docs)
        total = 0
        for idx, text in enumerate(docs):
            toks = tokenize(text)
            total += len(toks)
            self.doc_len.append(len(toks))
            tf = Counter(toks)
            for term, c in tf.items():
                self.postings[term].append((idx, c))
                self.df[term] = self.df.get(term, 0) + 1
        self.avgdl = total / max(self.N, 1)
        self.postings = dict(self.postings)
        return self

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def search(self, query: str, top_k: int = 600) -> list:
        """Return [(candidate_id, score), ...] for the top_k docs (only those matching query terms)."""
        q = tokenize(query)
        scores = defaultdict(float)
        for term in set(q):
            if term not in self.postings:
                continue
            idf = self._idf(term)
            for idx, tf in self.postings[term]:
                dl = self.doc_len[idx]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[idx] += idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        return [(self.ids[idx], s) for idx, s in ranked]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "BM25":
        with open(path, "rb") as f:
            return pickle.load(f)


def rrf_fuse(rank_lists: list, k: int = 60, top_k: int = 400) -> list:
    """Reciprocal Rank Fusion over several [(id, _score), ...] lists -> [(id, rrf), ...].
    Used ONLY for building the recall/rerank pool (not for the final precision score)."""
    rrf = defaultdict(float)
    for lst in rank_lists:
        for rank, (cid, _s) in enumerate(lst):
            rrf[cid] += 1.0 / (k + rank + 1)
    return sorted(rrf.items(), key=lambda x: (-x[1], x[0]))[:top_k]
