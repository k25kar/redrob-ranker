# Approach

How the ranker decides who belongs in the top 100, and why it's built this way.

## The problem

We're given 100,000 candidate profiles and one job description (a founding-team Senior AI Engineer who
owns search, ranking, and recommendation systems). The task is to return the 100 best-fit people,
ranked, the way a thoughtful recruiter would: by what someone has actually built, not by how many of
the right words appear on their profile.

Two things make it harder than a keyword search. First, plenty of profiles list the right skills
without any real work behind them, so surface matching ranks the wrong people. Second, the system has
to run cheaply and reproducibly: CPU only, no internet, no calls to an external model, finishing in a
few minutes on an ordinary machine.

## How we rank

Everyone is scored, then the list is trimmed. The flow:

1. **Filter out impossible profiles.** Some histories don't add up (more years of jobs than years of
   experience, dates that run backwards, and similar). These are dropped before scoring so they can
   never reach the top.
2. **Score the real work.** The main signal is the evidence in a person's career history of building
   production retrieval, ranking, and recommendation systems. Title, seniority, years of experience,
   and listed skills support that signal but don't drive it on their own.
3. **Widen the net and fuse it back in.** A lightweight keyword search runs alongside the scorer,
  then gets fused into the rerank pool so a strong candidate isn't overlooked just because their
  write-up is plainly worded.
4. **Account for reachability.** Someone who never replies to recruiters or hasn't been active in
   months is a weaker bet in practice, so their score is eased down. The effect is small and capped,
   so it nudges the order without burying a genuinely strong match.

The result is `outputs/submission.csv`: the top 100 with a one-line reason for each.

## Why this design

**Evidence over keywords.** The score leans on described, owned work rather than a skills list,
because a skills list is easy to pad and a track record is not. A "Recommendation Systems Engineer"
whose history is actually about churn models shouldn't outrank a "Software Engineer" who shipped
dense retrieval and learning-to-rank.

**No external model at ranking time.** Everything is plain Python and runs offline. That keeps the
system fast, reproducible, and free of the failures that come with depending on a hosted service
(rate limits, outages, nondeterminism). It also matches the challenge's compute rules.

**Interpretable by design.** Every score breaks down into named parts you can read and defend. There
is no black box to take on faith, which matters more than squeezing out a fragile last bit of
accuracy on a single job description.

## The signals

The score combines a handful of components, all computed from the profile itself:

- evidence of retrieval / ranking / recommendation work in the career history (the largest weight)
- how well the current and recent titles fit the role, with seniority taken into account
- years of experience, centered on the band the role asks for
- listed skills, used as corroboration and discounted when they look inconsistent with the rest of
  the profile
- education, as a minor factor
- reachability, applied as a small, capped adjustment at the end

All of the weights and thresholds live in `config.yaml`.

## Handling messy data

Real candidate pools contain profiles whose own fields contradict each other. The pipeline applies
plain consistency checks (tenure versus stated experience, sensible dates, and so on) and keeps those
profiles out of the top. The checks only act on contradictions inside a single profile, so they don't
penalize unusual but legitimate careers.

## Does it hold up

There's no answer key for this dataset, so the ranking was checked three ways and they agree:

- Reading the top of the list by hand. The highest-ranked people are senior ML, AI, NLP, search, and
  recommendation engineers with real ranking work in their histories.
- Watching the quality fall off with rank. A good ranker should get steadily weaker as you go down,
  and this one does. The standard metrics (NDCG, MAP, P@k) are in `metrics.py`.
- A separate recruiter-style rubric (`agents/recruiter_judge.md`, run via `scripts/judge.py`) grades
  the shortlist on its own as a second opinion. It's a development aid and never runs during ranking.

## Constraints and guarantees

- Offline at ranking time: no network, no GPU, no external model.
- Deterministic: ties break by candidate id and scores are rounded before sorting, so re-running
  produces the identical file.
- Input-checked: the dataset's hash is compared at startup, and on a mismatch the keyword index is
  rebuilt on the spot so the code still works on a different candidate pool.

## Honest limitations

The dataset is synthetic and the strong candidates are written in a fairly uniform style, so within
the very top of the list the exact order is less certain than the broad sorting into strong, middling,
and weak. The system is confident about who belongs near the top and who doesn't; it's less sure about
the precise ordering among a cluster of similar, strong people. Without ground-truth labels, that gap
can't be closed from our side, and we'd rather state it plainly than pretend otherwise.
