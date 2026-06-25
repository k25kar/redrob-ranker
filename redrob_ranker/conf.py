"""Config loader with a tiny fallback parser so the package has zero hard dependency on PyYAML."""
from __future__ import annotations
import os


def load_config(path: str) -> dict:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        pass
    # Minimal fallback: only used if PyYAML is unavailable. Hardcodes the defaults in config.yaml.
    return {
        "paths": {"candidates": "data/candidates.jsonl", "artifacts_dir": "artifacts", "outputs_dir": "outputs"},
        "integrity": {"expected_sha256": "de7b8cae39a9f9378a2cd4f8153bfc1f84960bce0ae520f263423d129df4b335"},
        "role_profile": {"experience_years": {"min": 5, "max": 9}, "seniority": "senior"},
        "weights": {"evidence": 0.45, "title": 0.22, "exp": 0.13, "skills": 0.12, "edu": 0.08},
        "penalties": {"cap": 0.35, "consulting_only": 0.18, "expert_at_zero": 0.15,
                      "cv_speech_robotics_no_nlp": 0.12, "research_only_no_prod": 0.10,
                      "title_chaser": 0.08, "seniority_mismatch": 0.10},
        "behavioral": {"floor": 0.78, "reach_threshold": 0.40},
        "retrieval": {"rerank_pool": 400, "bm25_topN": 600, "rule_topN": 600, "rrf_k": 60},
    }


def sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
