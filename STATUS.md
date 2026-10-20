# Project status (as of 2026-10-19 22:35 UTC)

↑ [Project README](README.md) · [TASKS](TASKS.md) · [INDEX](INDEX.md) · [Owner Q&A](OWNER_QA.md) · [Status history](docs/archive/STATUS_history.md)

One page, plain words. Every number comes from the [INDEX](INDEX.md) row named in brackets; INDEX is the only place
where numbers and claim status are kept. "(unverified)" means no second agent has checked it. Older dated updates are in
[docs/archive/STATUS_history.md](docs/archive/STATUS_history.md); ID prefixes are explained in the [README](README.md#id-prefixes-and-names).

**What the project is.** A tiny network (16 neurons) reads 10 random numbers and points to the second-largest; it is right
about 95% of the time. ARC's challenge: work out how often it is right from its weights alone, about as precisely as random
testing with the same computing (ARC: "competitive with sampling"; [Q&A 4.13](OWNER_QA.md#q4-13)). Behind it is a safety
goal: predicting failures too rare for testing to catch.

**Readable neurons.** In a larger language model, about 65% of neurons look like they have one main job, against about
21% of random directions (H-Q5, confirmed); readable parts alone were not enough to calculate behaviour.

**The main challenge: two candidates, ARC acceptance unknown for both.**
- *Line walks* (a fixed, evenly spread set of lines; each line exact from the weights): at the full 10 steps about equal
  to random testing at small budgets, and about half its error from about a million runs per model on a sealed,
  never-used panel (H-E17, confirmed; T431). On wider networks (24 and 32 neurons) it wins 2-6x even at small budgets
  (T423, interim). It stays a quadrature-type method: no bound from the weights on its averaging error exists in any
  family tried (T449, verified by T451). Checks on noisy copies and the target (T434) and on other line sets (T442) run.
- *Gap-direction walks with five declared approximations:* about 2.6x random testing's error on a sealed panel (H-L1),
  inside ARC's own "rough match" example; the approximations were tuned on sampled data.

**Rare mistakes (ARC's long-term goal).**
- *Estimating:* within about 2x down to one in 10 billion on ordinary networks (H-R1, confirmed). On the challenge
  network an improved approximation passed its first test and a sealed test runs (T370). Sampling focused on the most
  likely failure points estimates the studied mistake (about 1.5 in 10 billion) with about 1e5x less computing than plain
  random testing (T436); on 15 blind points it is within 2x on all 15, but its error bars are honest on only 12 (T453,
  T454); an improved version scored on 15 fresh points is within 3% with combined-error z scores within 2 on all 15 (T456; saved arithmetic checked by T461, general interval calibration still open).
- *Rating safe use* ("if the top numbers differ by at least X, the error rate is Y"): works on some networks, partly on
  the challenge network (T401, T407); on a wider network it works for the rarer conditions instead (T427, interim).
- *Locating and repairing:* a 12-weight edit makes the studied mistake about 17x rarer with almost no accuracy cost while
  random edits fail (H-R3, confirmed); removing all such mistakes is not possible, because repairs turn rare mistakes
  into common ones (T411).
- *Guaranteeing an upper limit:* sampling puts the studied mistake at about 1.6 in 10 billion, at most 4.2 in 10 billion
  with very high confidence (T400). For one section of inputs a mathematical guarantee gives at most 1.4 in 100 billion
  (T416, provisional until the geometry audit T420). The section holding all observed mistakes has about 2 million
  pieces and takes about 5 days of CPU to charge (T398, T400; speed-up T452). The late sections looked blocked, but the
  blocker was the splitting rule, not the bounds: a better rule projects about 2e6-2e7 pieces per section (T418), and a
  rigorous version runs (T455). A guarantee over all inputs may therefore be feasible.

**Transfer.** Other tasks on the same network shape: 24 of 29 candidates passed the quick screen (T419). Bigger networks:
16 neurons cannot do 20 numbers, 32 can (T421). A mini language model: two small languages and 36 baseline models
(T433); attention that keeps our exact methods usable matches standard attention on the bracket language (T444);
estimating, rare-mistake and guarantee streams run (T445-T447).

**Novelty.** Most of our tools apply known verification and line-sampling work; likely new are the rigorous Gaussian
masses of 10-D input cells, sampling-free rare-failure estimates near one in 10 billion, and the sealed strict result
([lit/novelty_review.md](lit/novelty_review.md); [Q&A 4.15](OWNER_QA.md#q4-15)).

**Ideas that did not work** (each in INDEX): a bound from the weights on the line walks' error (T449, T451), rounded-value
networks (T352), mixing exact regions with tracing (T46), listing mistake patterns (T413), enumerating regions (T373,
T390), a GPU checker (T399), algebraic rewrites (T291, T3), longer training of a small test rung (T111), a deductive
method for the all-at-once network (T408). A bug re-check of the small-rung calculator changed no verdicts (T443).

**New checks, 20 October:** CPU controls exposed a floating-point defect in T455’s proposed box-mass bound and a reliability limitation of the importance-sampling warning gate. Both were independently checked; local repairs/reporting alternatives are recorded in T462/T463. The full search and trained uncertainty remain unverified. These are scoped findings, not a reversal of H-C1 or the saved trained-panel observations.

**Coordination, 20 October:** Codex is acting DRI during Claude’s credit outage. The live Codex roster initially had no workers; two bounded reviews and an independently checked saved-result audit are complete. Existing Claude rows do not establish live sessions. [Takeover decisions and next priorities](docs/DRI_TAKEOVER_2026-10-20.md).

**Recorded assignments:** see [TASKS.md](TASKS.md#in-progress) (owners and sessions); the next DRI's queue is in
[DRI_HANDBOOK.md section 12](DRI_HANDBOOK.md).

**Documents:** what more money or compute would solve in [docs/COMPUTE_FRONTIER.md](docs/COMPUTE_FRONTIER.md).
**Standing rules** (no ARC contact, no Fable, CPU-first, one home per fact) are in [AGENTS.md](AGENTS.md#standing-rules).
