# Recruiter-Judge Report — scoring our model's shortlist

**Judge:** `agents/recruiter_judge.md` rubric, applied independently to the candidates' full
profiles (career-history descriptions, signals), *without* reusing the ranking model's logic.
**Scope:** top-12 of `submission.csv` judged in detail, plus 5 controls (2 honeypots, a mid-rank,
a low-rank, a junior) to test the gradient and the gate.

---

## Top-12 (our model's best picks) — tier per the rubric

| model rank | candidate | title | yoe | tier | interview | basis (quoted from profile) |
|---|---|---|---|---|---|---|
| 1 | CAND_0018499 | Sr MLE | 7.2 | **3** | Yes | "hybrid retrieval BM25 + dense (BGE, FAISS HNSW), LLM re-ranker top-50, fallback LTR"; eval harness; resp 0.61, active, open |
| 2 | CAND_0081846 | Lead AI Eng | 6.7 | **3** | Yes | "rebuilt candidate-JD matching 0.72→0.91 NDCG@10"; semantic search migration; product cos (Razorpay/Paytm) |
| 3 | CAND_0008425 | Sr NLP Eng | 7.8 | **3** | Yes | semantic search 35M items; "Pinecone retrieval → LTR re-scoring (XGBoost)"; Ola/Amazon |
| 4 | CAND_0055905 | Sr MLE | 8.1 | **3** | Yes | "migration from keyword ranking to learning-to-rank with behavioral signals"; resp 0.87 |
| 5 | CAND_0070398 | MLE | 7.2 | **3** | Yes | semantic search + FAISS; "ranking models … offline-online correlation"; Uber/Mad Street Den |
| 6 | CAND_0077337 | Staff MLE | 7.0 | **3** | Yes | hybrid retrieval; embedding-search migration with A/B; resp 0.95 |
| 7 | CAND_0071974 | Sr AI Eng | 7.8 | **3** | Yes | "end-to-end ranking pipeline … Pinecone → LTR (XGBoost) → behavioral signals"; Netflix/Meta |
| 8 | CAND_0046525 | Sr MLE | 6.1 | **3** | Yes | hybrid BM25+dense; embedding migration + A/B; LinkedIn; resp 0.88 |
| 9 | CAND_0066376 | Applied MLE | 5.7 | **3** | Yes | content recsys 10M users; semantic search FAISS; Dream11/Salesforce |
| 10 | CAND_0028793 | Search Eng | 7.2 | **3** | Yes | semantic search FAISS; ranking models; offline-online correlation; Google/Meesho |
| 11 | CAND_0086022 | Sr Applied Sci | 5.3 | **3** | Yes | RAG ranking pipeline; embedding-search migration; LTR; Sarvam/Uber |
| 12 | CAND_0088025 | Staff MLE | 8.6 | **3** | Yes | end-to-end ranking; Pinecone; LTR; 8.6y |

**Every one of the top-12 shows specific, described, production retrieval/ranking/recsys work, right seniority, and is reachable.** No keyword-stuffers, no domain mismatches, no honeypots.

## Controls — gradient & gate

| control (model's treatment) | title | finding | tier |
|---|---|---|---|
| CAND_0003582 (**excluded** from top-100) | Mobile Developer | 3 "expert" skills at 0 months (honeypot signature); described work is QA/frontend/DevOps, **zero AI/ML** | **0** + 🚩honeypot |
| CAND_0007353 (**excluded**, hard gate) | Frontend Engineer | tenure-sum **251mo** vs 119mo experience = impossible; frontend/devops only | **0** + 🚩honeypot |
| CAND_0010257 (model rank ~85) | Sr Data Scientist | some ranking/recsys (XGBoost ranking, content recsys) mixed with churn/MLOps; reachable | **2** |
| CAND_0095884 (model rank ~300) | Sr SWE (ML) | "recommendation-style features … lighter than ranking systems at FAANG"; 3.8y; not open; resp 0.39 | **1** |
| CAND_0001151 (model rank ~318) | Junior ML Engineer | generic ML (forecasting, fraud); junior; no real IR; not open | **1** (borderline 0) |

---

## Aggregate verdict

- **Tier distribution (top-12):** 3→12, 2→0, 1→0, 0→0.
- **Precision@10 (Tier-3):** **1.00**  ·  **Hire-rate@10 (Tier≥2):** **1.00**  ·  Hire-rate@12: **1.00**
- **NDCG@10 (judge tiers vs model order):** **1.00** (all top-12 are Tier 3).
- **Honeypot-rate in top-100:** **0** — both honeypots were correctly kept out (one by the hard gate, one by low evidence + soft penalty). Well under the 10% disqualifier.
- **Gradient (rule behaves like relevance):** Tier **3** at ranks 1–12 → Tier **2** at ~85 → Tier **1** at ~300–318. Monotonic. This is what a trustworthy ranker produces.

**Recruiter's bottom line:** *This is a shortlist I would forward to a hiring manager.* The top dozen are all people I'd schedule this week; the honeypots and the generic/junior/under-experienced profiles are correctly far down or excluded.

---

## Independence caveats (read these — they matter for Stage 5)

1. **I am an LLM judge reading the same synthetic profiles**, so this is a strong *second opinion*, not the hidden ground truth. It mainly proves the ranker isn't doing anything a careful recruiter would disagree with — it does **not** prove we match the organizers' exact label process.
2. **The top-12 share near-identical templated prose** (the same "BM25 + dense retrieval (BGE, FAISS HNSW)…" paragraph recurs). Every one earns Tier 3 *on the evidence as written*, but a real recruiter would interview to disambiguate near-clones. Consequently **NDCG@10 = 1.00 is partly trivial** — when everyone in the top is Tier 3, the metric can't reward *within-tier* ordering, which is exactly the "ordering inside the elite cohort is uncertain" limitation we already documented.
3. The judge **confirms the cohort and the gradient**, not the fine ordering. That is the honest extent of what any second opinion can establish without the answer key.

---

## Reproduce / scale this (with a free LLM, offline — evaluation only)
```bash
export JUDGE_API_BASE=https://api.groq.com/openai/v1      # or OpenRouter / NVIDIA NIM
export JUDGE_API_KEY=...                                   # your free key
export JUDGE_MODEL=llama-3.3-70b-versatile                 # or deepseek/deepseek-r1:free
python scripts/judge.py --candidates data/candidates.jsonl --submission outputs/submission.csv --n 50
# (LLM is allowed here: this scores the OUTPUT offline; it is NOT the ranking step.)
```
