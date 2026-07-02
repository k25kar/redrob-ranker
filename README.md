# Redrob Ranker

Picks the best 100 people out of 100,000 for the Redrob "Senior AI Engineer (Founding Team)" role.
It judges candidates on the work they have actually done, not on how many of the right terms they list.

It runs on a normal laptop in under a minute. No GPU, no internet, no API keys. Run it twice and you
get the exact same file both times.

## How it works

```
        100,000 candidates
                 |
   1.  drop profiles whose history is impossible
                 |   (20 years of jobs in a 9-year career, etc.)
                 |
   2.  score the real search / ranking / recsys
       work in each person's history
                 |   (title, seniority, experience, skills back it up)
                 |
   3.  ease down people who never reply, went quiet
       months ago, or are hard to reach or place
                 |   (availability + location + verified identity)
                 |
   4.  break near-ties by the qualities the role values:
       evaluation rigor, seniority, ownership, impact
                 |
                 v
        top 100  ->  submission.csv
```

Step 2 does most of the work, and it reads carefully: an identical paragraph pasted into three jobs
counts as evidence once, not three times, and plain-language descriptions of ranking work ("how the
most relevant results appear for each user's intent") score alongside buzzword-dense ones.
Step 1 keeps profiles whose own data does not add up out of the top.
Step 3 reflects that a strong profile you cannot actually reach or place is less useful to a recruiter.
Step 4 separates strong candidates the fit score sees as nearly equal, using the signals the job
description prioritises. Steps 1, 3, and 4 are small, bounded adjustments.

The challenge forbids calling an external model while ranking, so we do not. Everything here is plain
Python. The longer write-up of the approach is in `PLAN.md`.

## What's in here

```
redrob_ranker/          the code
  schema.py             reads the candidates; -1 means "no data", not "worst score"
  features.py           the signals: evidence, title, seniority, experience, skills, education
  gates.py              the impossible-profile filter, plus soft penalties
  behavioral.py         availability + location/notice + identity verification (small, capped)
  quality.py            near-tie refinement by role-relevant signals (eval rigor, seniority, impact)
  scoring.py            combines the pieces into one score
  conf.py               reads config.yaml

scripts/
  rank.py               the ranking step: writes outputs/submission.csv  (this is all you need)
  verify.py             sanity checks: dataset hash, row count, output format

config.yaml             every knob in one place (weights, thresholds)
data/                   put candidates.jsonl here (not committed, ~487 MB)
outputs/                submission.csv
```

## Running it

```bash
pip install -r requirements.txt        # only PyYAML; the ranking itself needs nothing else

# download candidates.jsonl from the challenge, then point the code at it
ln -s /path/to/candidates.jsonl data/candidates.jsonl

python scripts/rank.py   --candidates data/candidates.jsonl   # ~40s, writes outputs/submission.csv
python scripts/verify.py --candidates data/candidates.jsonl --submission outputs/submission.csv
```

The answer is `outputs/submission.csv`: 100 rows of candidate_id, rank, score, and a short reason.
No pre-build step is needed; the ranker uses no index or external model.

## Guarantees

It never touches the network, a GPU, or an external model while ranking.

It is deterministic. Ties break by candidate_id and scores are rounded before sorting, so you get the
same file every run.

It checks its input: the dataset's SHA-256 is compared at startup and warns if the pool differs from
the one it was configured for. The rule score works on any candidate pool, so nothing else is needed.
