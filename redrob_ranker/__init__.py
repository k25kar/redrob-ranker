"""Redrob candidate ranker.

An interpretable, embedding-free scorer that ranks candidates for a job description by the
evidence of relevant work in their history. The ranking step is deterministic, CPU-only, and
makes no network calls. See PLAN.md for the approach.
"""
__version__ = "1.0.0"
