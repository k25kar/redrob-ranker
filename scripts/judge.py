#!/usr/bin/env python3
"""Recruiter-judge harness — scores our model's shortlist with an independent LLM judge.

EVALUATION ONLY. This is NOT part of rank.py and never runs at ranking time. Using an LLM here is
allowed because it judges the *output* offline; it does not produce the submission.

The judge persona + rubric live in agents/recruiter_judge.md (loaded as the system prompt).
Works with any OpenAI-compatible endpoint via env vars (all free tiers supported):
    JUDGE_API_BASE  e.g. https://api.groq.com/openai/v1
                    or   https://openrouter.ai/api/v1
                    or   https://integrate.api.nvidia.com/v1
    JUDGE_API_KEY   your free key
    JUDGE_MODEL     e.g. llama-3.3-70b-versatile  /  deepseek/deepseek-r1:free  /  meta/llama-3.1-70b-instruct

Usage:
    python scripts/judge.py --candidates data/candidates.jsonl --submission outputs/submission.csv --n 50
    python scripts/judge.py ... --dry-run        # emit prompts only, no API calls (free, offline)
"""
import argparse, csv, json, math, os, sys, time, hashlib

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUBRIC = os.path.join(HERE, "agents", "recruiter_judge.md")
CACHE = os.path.join(HERE, "outputs", ".judge_cache")


def load_submission(path):
    return list(csv.DictReader(open(path)))


def index_candidates(path, wanted):
    by_id = {}
    for line in open(path):
        if not line.strip():
            continue
        c = json.loads(line)
        if c["candidate_id"] in wanted:
            by_id[c["candidate_id"]] = c
    return by_id


def candidate_payload(c):
    """Compact, faithful JSON for the judge (no truncation of descriptions — they carry the evidence)."""
    p = c["profile"]; rs = c["redrob_signals"]
    return {
        "candidate_id": c["candidate_id"],
        "headline": p.get("headline"), "summary": p.get("summary"),
        "current_title": p.get("current_title"), "years_of_experience": p.get("years_of_experience"),
        "career_history": [{"title": h["title"], "company": h["company"],
                            "duration_months": h.get("duration_months"), "industry": h.get("industry"),
                            "description": h.get("description")} for h in c["career_history"]],
        "skills": [{"name": s["name"], "proficiency": s["proficiency"],
                    "endorsements": s.get("endorsements"), "duration_months": s.get("duration_months")}
                   for s in c.get("skills", [])],
        "education": c.get("education", []),
        "signals": {k: rs.get(k) for k in ("recruiter_response_rate", "last_active_date",
                    "open_to_work_flag", "interview_completion_rate", "notice_period_days",
                    "github_activity_score")},
    }


def call_llm(system, user, base, key, model, retries=4):
    import urllib.request
    body = json.dumps({"model": model, "temperature": 0.0,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read())
            return out["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def parse_json(txt):
    s, e = txt.find("{"), txt.rfind("}")
    return json.loads(txt[s:e + 1]) if s >= 0 and e > s else None


def ndcg(tiers_in_order, k):
    dcg = sum((2 ** t - 1) / math.log2(i + 2) for i, t in enumerate(tiers_in_order[:k]))
    idcg = sum((2 ** t - 1) / math.log2(i + 2) for i, t in enumerate(sorted(tiers_in_order, reverse=True)[:k]))
    return dcg / idcg if idcg else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--n", type=int, default=50, help="judge the top-N rows")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="outputs/judge_report.md")
    a = ap.parse_args()

    system = open(RUBRIC).read()
    rows = load_submission(a.submission)[: a.n]
    ids = [r["candidate_id"] for r in rows]
    cands = index_candidates(a.candidates, set(ids))
    os.makedirs(CACHE, exist_ok=True)

    base, key, model = os.getenv("JUDGE_API_BASE"), os.getenv("JUDGE_API_KEY"), os.getenv("JUDGE_MODEL")
    if a.dry_run:
        ex = json.dumps(candidate_payload(cands[ids[0]]), indent=2)[:1500]
        print("DRY RUN — system prompt = agents/recruiter_judge.md")
        print(f"would judge {len(ids)} candidates via {model} @ {base}")
        print("example user payload (truncated):\n", ex)
        return
    assert base and key and model, "set JUDGE_API_BASE / JUDGE_API_KEY / JUDGE_MODEL (free tier ok)"

    results = []
    for rank, cid in enumerate(ids, 1):
        cpath = os.path.join(CACHE, cid + ".json")
        if os.path.exists(cpath):
            results.append(json.load(open(cpath))); continue
        payload = candidate_payload(cands[cid])
        user = ("Score this candidate per the rubric. Return ONLY the per-candidate JSON object.\n\n"
                + json.dumps(payload, ensure_ascii=False))
        txt = call_llm(system, user, base, key, model)
        obj = parse_json(txt) or {"candidate_id": cid, "tier": 0, "would_interview": False,
                                  "one_line": "parse_error", "concerns": ["judge parse error"]}
        obj["model_rank"] = rank
        json.dump(obj, open(cpath, "w"))
        results.append(obj)
        print(f"judged {rank}/{len(ids)} {cid} -> tier {obj.get('tier')}")

    tiers = [int(r.get("tier", 0)) for r in results]
    dist = {t: tiers.count(t) for t in (3, 2, 1, 0)}
    p10 = sum(1 for t in tiers[:10] if t == 3) / 10
    hire10 = sum(1 for t in tiers[:10] if t >= 2) / 10
    hire20 = sum(1 for t in tiers[:20] if t >= 2) / min(20, len(tiers))
    hp = sum(1 for r in results if r.get("honeypot_flag"))

    with open(a.out, "w") as f:
        f.write("# Recruiter-Judge Report (independent LLM judge)\n\n")
        f.write(f"- model: `{model}`  | judged top-{len(results)}\n")
        f.write(f"- tier distribution (3/2/1/0): {dist[3]}/{dist[2]}/{dist[1]}/{dist[0]}\n")
        f.write(f"- Precision@10 (tier-3): {p10:.2f}  | hire-rate@10 (tier>=2): {hire10:.2f}  | hire-rate@20: {hire20:.2f}\n")
        f.write(f"- NDCG@10: {ndcg(tiers,10):.3f}  | NDCG@50: {ndcg(tiers,50):.3f}\n")
        f.write(f"- honeypot flags in judged set: {hp}\n\n")
        f.write("| rank | candidate | tier | interview | one-line |\n|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r.get('model_rank')} | {r['candidate_id']} | {r.get('tier')} | "
                    f"{'Y' if r.get('would_interview') else 'N'} | {str(r.get('one_line','')).replace('|','/')[:140]} |\n")
    print(f"\nwrote {a.out}  | tiers {dist} | P@10 {p10:.2f} hire@10 {hire10:.2f} NDCG@10 {ndcg(tiers,10):.3f}")


if __name__ == "__main__":
    main()
