"""Redrob candidate ranker — embedding-free rule scorer + BM25 retrieval (+ cross-encoder rerank, rung 3).

See FINAL_IMPLEMENTATION_PLAN.md §0 for the authoritative design.
The ranking step is deterministic, CPU-only, and makes no network calls.
"""
__version__ = "0.2.0"
