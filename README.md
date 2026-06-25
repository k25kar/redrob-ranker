# Redrob Ranker

Picks the best 100 people out of 100,000 for the Redrob "Senior AI Engineer (Founding Team)" role.
It judges candidates on the work they've actually done, not on how many buzzwords they list.

It runs on a normal laptop in about 17 seconds. No GPU, no internet, no API keys. Run it twice and
you get the exact same file both times.

## How it works

```
        100,000 candidates
                 |
   1.  throw out the impossible fakes
                 |   
                 |   (20 years of jobs in a 9-year career, etc.)
                 |
   2.  score the real search / ranking / recsys
       work in each person's history
                 |   
                 |   (title, seniority, experience, skills back it up)
                 |
   3.  widen the net with a keyword search
       so nobody strong slips past
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

Step 2 does most of the work. Steps 1, 3, and 4 keep it honest: step 1 stops fake profiles getting
near the top, step 3 makes sure a good person isn't missed just because their wording is plain, and
step 4 reflects that a great profile you can't actually reach isn't much use to a recruiter.

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
  rank.py               the main thing: writes outputs/submission.csv
  verify.py             proof: file hash, row count, no honeypots in the top 100
  diagnose.py           digging: what BM25 adds, where good people sit, score spread
  ablate_evidence.py    a stress test: real signal, or just the template?
  audit.py              checks the evidence scorer isn't too easily fooled
  judge.py              optional: an AI recruiter grades the output (I use it to sanity-check)

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

## Does it actually work?

There's no answer key, so I checked it three ways and they agree.

I read the top 20 profiles myself. They're all senior ML, AI, NLP, search, and recommendation
engineers with real ranking work in their history (see `outputs/top20_profiles.txt`). None of the
keyword-stuffer traps the dataset plants.

The quality drops as you go down the list, which is what a real ranking should do: roughly 19 of the
top 20 are people I'd interview, about 13 in 20 near rank 90, and about 5 in 20 near rank 300.

I also had an AI recruiter grade the output on its own (`agents/recruiter_judge.md`,
`outputs/judge_report.md`). It put the top picks in the top tier and caught the fake profiles. That
judge is just my own check. It never runs as part of the ranking.

Fake "honeypot" profiles in the top 100: zero. The challenge disqualifies you above ten percent.

## A few promises it keeps

It never touches the network, a GPU, or an AI service while ranking. The challenge requires that.

It's deterministic. Ties break by candidate_id, scores are rounded before sorting, so you get the
same file every run.

It checks the data hasn't changed: the dataset's SHA-256 is compared at startup, and if it doesn't
match it rebuilds the keyword index on the spot so the code still works on a different candidate pool.
