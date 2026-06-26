# Approach

How the ranker decides who belongs in the top 100, and why it is built this way.

## The problem

We are given 100,000 candidate profiles and one job description (a founding-team Senior AI Engineer
who owns search, ranking, and recommendation systems). The task is to return the 100 best-fit people,
ranked, the way a thoughtful recruiter would: by what someone has actually built, not by how many of
the right words appear on their profile.

Two things make it harder than a keyword search. First, many profiles list the right skills without
any real work behind them, so surface matching ranks the wrong people. Second, the system has to run
cheaply and reproducibly: CPU only, no internet, no external model, finishing in a few minutes on an
ordinary machine.

## How we rank

Everyone is scored, then the list is trimmed. The flow:

1. **Filter out impossible profiles.** Some histories do not add up (more years of jobs than years of
   experience, dates that run backwards, and similar). These are dropped before scoring so they can
   never reach the top.
2. **Score the real work.** The main signal is the evidence in a person's career history of building
   production retrieval, ranking, and recommendation systems. Title, seniority, years of experience,
   and listed skills support that signal but do not drive it on their own.
3. **Account for reachability, location, and trust.** Someone who never replies to recruiters or has
   not been active in months is a weaker bet in practice; the role is a Pune/Noida hybrid in India;
   and a verified, reachable profile is a stronger lead than an unverified one. Availability, location,
   and identity verification are applied as small, capped adjustments that nudge the order between
   close candidates without burying a genuinely strong match.
4. **Refine the order within near-ties.** When two strong candidates have nearly identical fit scores,
   the tie is broken using the qualities the job description prioritises: evaluation rigor, seniority,
   ownership, and measurable impact. This carries a small weight, so it only reorders candidates the
   fit score already treats as interchangeable and never lifts a weaker candidate across tiers.

The result is `outputs/submission.csv`: the top 100 with a one-line reason for each.

## Why this design

**Evidence over keywords.** The score leans on described, owned work rather than a skills list,
because a skills list is easy to pad and a track record is not. A "Recommendation Systems Engineer"
whose history is actually about churn models should not outrank a "Software Engineer" who shipped
dense retrieval and learning-to-rank.

**No external model at ranking time.** Everything is plain Python and runs offline. That keeps the
system fast, reproducible, and free of the failures that come with depending on a hosted service. It
also matches the challenge's compute rules.

**Interpretable by design.** Every score breaks down into named parts that can be read and explained.
There is no black box to take on faith, which matters more than chasing a fragile last fraction of
accuracy on a single job description.

## The signals

The score combines a handful of components, all computed from the profile itself:

- evidence of retrieval / ranking / recommendation work in the career history (the largest weight)
- how well the current and recent titles fit the role, with seniority taken into account
- years of experience, centered on the band the role asks for
- listed skills, used as corroboration and discounted when they look inconsistent with the rest of the profile
- education, as a minor factor
- reachability, location, and identity verification, applied as small, capped adjustments
- among near-ties, the role-relevant quality signals above

All weights and thresholds live in `config.yaml`.

## What we deliberately do not use

- **A learned ranking model.** Training one needs many example roles with known-good rankings; here
  there is a single role, so a learned model has too little to learn from. A transparent weighted
  score is both stronger in this setting and easier to explain.
- **Embeddings or any external model at ranking time.** The challenge forbids it, and similarity-based
  scoring also tends to reward profiles that merely echo the job description.
- **Expected salary.** It does not vary with experience in this data and the job description states no
  band, so it carries no usable signal.
- **Candidate name, languages, and degree field.** Deliberately ignored to keep scoring focused on
  demonstrated ability rather than background.

## Handling messy data

Real candidate pools contain profiles whose own fields contradict each other. The pipeline applies
plain consistency checks (tenure versus stated experience, sensible dates, and so on) and keeps those
profiles out of the top. The checks act only on contradictions inside a single profile, so they do not
penalise unusual but legitimate careers. Self-listed skills are discounted when they are not supported
by the described work, and reasons are generated only from facts in the profile.

## Constraints and guarantees

- Offline at ranking time: no network, no GPU, no external model.
- Deterministic: ties break by candidate id and scores are rounded before sorting, so re-running
  produces the identical file.
- Input-checked: the dataset's hash is compared at startup and warns if the pool differs. The rule
  score works on any candidate pool.

## A note on the top of the list

The strongest candidates are written in a fairly uniform style, so the very top few are close peers.
We order them by the qualities the role prioritises (evaluation rigor, seniority, ownership, and
measurable impact), which is the policy a recruiter would apply. The system is confident about who
belongs near the top and who does not; among a handful of near-equal peers, the exact one-against-two
order is inherently a judgment call.
