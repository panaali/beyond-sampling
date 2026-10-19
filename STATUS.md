# Project status (as of 2026-10-19 16:40 UTC)

↑ [Project README](README.md) · [TASKS](TASKS.md) · [INDEX](INDEX.md) · [Owner Q&A](OWNER_QA.md) · [Status history](docs/archive/STATUS_history.md)

One page, plain words. Every number comes from the [INDEX](INDEX.md) row named in brackets; INDEX is the only place
where numbers and claim status are kept. "(unverified)" means no second agent has checked it. Older dated updates are in
[docs/archive/STATUS_history.md](docs/archive/STATUS_history.md); ID prefixes are explained in the [README](README.md#id-prefixes-and-names).

**What the project is.** A tiny network (16 neurons) reads 10 random numbers and points to the second-largest; it is right
about 95% of the time. ARC's challenge: work out how often it is right from its weights alone, more precisely than random
testing with the same computing. Behind it is a safety goal: predicting failures too rare for testing to catch.

**Readable neurons.** In a larger language model, about 65% of neurons look like they have one main job, against about
21% of random directions (H-Q5, confirmed); readable parts alone were not enough to calculate behaviour.

**The main challenge: open.** The best weight-based calculation has about 2.6x random testing's error (lenient pass,
strict fail; H-L1, T317f). Exact tracing along lines through input space beats random testing on a 5-step version
(H-E16, confirmed) and on a network that sees all numbers at once (H-A3, confirmed), but only ties at the full 10 steps;
nothing qualifies for the sealed final test. New and unverified: on wider networks (24 and 32 neurons) the same frozen
method beats random testing 2-6x (T423, interim). Whether more computing helps at 10 steps is being tested (T429).

**Rare mistakes (ARC's long-term goal).**
- *Estimating:* within about 2x down to one in 10 billion on ordinary networks (H-R1, confirmed); about 80% right in a
  blind test on near-equal inputs, without knowing where it fails (T403); noise in the weights does not hurt (T388, T404).
  On the challenge network the deepest estimates are still too low; an improved approximation fixes most of it (T370).
- *Rating safe use* ("if the top numbers differ by at least X, the error rate is Y"): works on some networks, partly on
  the challenge network (T401, T407); on a wider network it works for the rarer conditions instead (T427, interim).
- *Locating and repairing:* the responsible weights are found (H-R2) and a 12-weight edit makes the mistake about 17x
  rarer with almost no accuracy cost while random edits fail (H-R3, confirmed). Removing all such mistakes by small edits
  is not possible: training already optimised the common mistakes, so repairs convert rare into common ones (T411).
- *Guaranteeing an upper limit:* the true rate of the studied mistake is about 1.6 in 10 billion, at most 4.2 in 10
  billion with very high confidence (sampling-based, T400). For one section of inputs a mathematical guarantee is nearly
  done: all ~260k safety certificates verify exactly (T400), the fast rigorous calculator is confirmed (H-C1), the
  geometry is under audit (T420). The section holding all observed mistakes is much costlier to split (T398, interim).
  A guarantee over all inputs is out of reach with our methods (T398, T414); a final library check runs (T418).

**Ideas that did not work** (each in INDEX): rounded-value networks (T352), mixing exact regions with tracing (T46),
listing mistake patterns (T413), enumerating regions (T373, T390), a GPU checker (T399), algebraic rewrites (T291, T3),
longer training of a small test rung (T111), a deductive method for the all-at-once network (T408).

**Running now** (owners in [TASKS.md](TASKS.md#in-progress)): transfer to about 10 other tasks (T419: 24 of 29 candidates
passed the quick screen), larger networks with five method tests (T421, T423-T427), the guarantee work (T398, T400/T416,
T418, T420), more-compute test for tracing (T429), the approximation fix (T370), and ladders (T86, T46, T392).

**Documents:** novelty versus prior work in [lit/novelty_review.md](lit/novelty_review.md); what more money or compute
would solve in [docs/COMPUTE_FRONTIER.md](docs/COMPUTE_FRONTIER.md). **Standing rules** (no ARC contact, no Fable,
CPU-first, tutorial paused, one home per fact) are in [AGENTS.md](AGENTS.md#standing-rules).
