# Redrob Ranker

Picks the best 100 people out of 100,000 for the Redrob "Senior AI Engineer (Founding Team)" role.
It judges candidates on the work they've actually done, not on how many buzzwords they list.

It runs on a normal laptop in about 17 seconds. No GPU, no internet, no API keys. Run it twice and you get the exact same file both times.

## How it works

```
        100,000 candidates
                 |
   1.  drop profiles whose history is impossible
                 |   
                 |   (20 years of jobs in a 9-year career, etc.)
                 |
   2.  score the real search / ranking / recsys
       work in each person's history
                 |   
                 |   (title, seniority, experience, skills back it up)
                 |
     3.  widen the net with a keyword search
       and fuse it into the rerank pool
                 |
                 |
                 |
   4.  ease down people who never reply
       or went quiet months ago
                 |
                 |
                 |
                 v
        top 100  ->  submission.csv
```

Step 2 does most of the work. Steps 1, 3, and 4 keep it honest: step 1 keeps profiles whose own data
doesn't add up out of the top, step 3 broadens recall and feeds a deterministic fusion stage so a good
person isn't missed just because their wording is plain, and step 4 reflects that a great profile you
can't actually reach isn't much use to a recruiter.

The challenge bans calling an AI service while ranking, so we don't. Everything here is plain Python.
If you want the long version of why it's built this way, read `/PLAN.md`.

## What's in here

```
redrob_ranker/          the code
  schema.py             reads the candidates; -1 means "no data", not "worst score"
  features.py           the signals: evidence, title, seniority, experience, skills, education
  gates.py              the impossible-profile filter, plus soft penalties (consulting, etc.)
  behavioral.py         how reachable someone is (small effect, capped)
  scoring.py            puts the pieces together into one score
  retrieval.py          a small BM25 keyword search, no outside libraries
  metrics.py            NDCG@10/50, MAP, P@10, for checking quality offline
  conf.py               reads config.yaml

scripts/
  build_artifacts.py    run once first: builds the keyword index
  rank.py               the ranking step: writes outputs/submission.csv using rule+BM25 fusion
  verify.py             sanity checks: dataset hash, row count, output format
  diagnose.py           analysis: what BM25 adds, where strong candidates sit, score spread
  ablate_evidence.py    a stress test: is the score driven by real signal or just wording?
  audit.py              checks the evidence scorer isn't too easily fooled
  judge.py              optional: a recruiter-style rubric grades the output as a cross-check

agents/recruiter_judge.md   the AI recruiter's instructions, used by judge.py
config.yaml                 every knob in one place (weights, thresholds)
data/                       drop candidates.jsonl here (too big to commit, 487 MB)
artifacts/                  the generated index lives here (rebuilt by the script)
outputs/                    submission.csv and the analysis files
```

## Running it

```bash
pip install -r requirements.txt        # only PyYAML; the ranking itself needs nothing else

# download candidates.jsonl from the challenge, then point the code at it
ln -s /path/to/candidates.jsonl data/candidates.jsonl

python scripts/build_artifacts.py --candidates data/candidates.jsonl   # once, ~15s
python scripts/rank.py            --candidates data/candidates.jsonl   # ~17s, writes the submission
python scripts/verify.py --candidates data/candidates.jsonl --submission outputs/submission.csv
```

The answer is `outputs/submission.csv`: 100 rows of candidate_id, rank, score, and a short reason.

## Validation

The dataset doesn't ship ground-truth labels, so the ranking was checked three ways, and they agree.

Manual review of the top of the list: the highest-ranked candidates are senior ML, AI, NLP, search,
and recommendation engineers whose career histories describe building production retrieval and
ranking systems, rather than just listing the right terms.

Relevance gradient: candidate quality declines steadily as rank increases, which is the expected
behaviour of a well-calibrated ranker. The standard ranking metrics (NDCG, MAP, P@k) live in
`metrics.py`, and `scripts/diagnose.py` reproduces the gradient analysis.

Independent second opinion: a separate recruiter-style rubric (`agents/recruiter_judge.md`, run via
`scripts/judge.py`) re-scores the shortlist offline as a cross-check. It is a development aid only and
never runs during ranking.

## Guarantees

It never touches the network, a GPU, or an external model while ranking.

It's deterministic. Ties break by candidate_id and scores are rounded before sorting, so you get the
same file every run.

It checks its input: the dataset's SHA-256 is compared at startup, and on a mismatch it rebuilds the
keyword index on the spot so the code still works on a different candidate pool.
