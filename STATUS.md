# Project status (2026-10-17)

↑ [Project README](README.md) · [TASKS](TASKS.md) · [INDEX](INDEX.md) · [RESULTS](RESULTS.md) · [FINDINGS](m16/FINDINGS.md) · [Approaches](m16/approaches/README.md)

The registry, hypothesis ledger and every number with its source are in [`INDEX.md`](INDEX.md). This page summarises them.

## In plain words

**What we set out to check.** A Goodfire essay ("Neural Networks Are Already Clean") claims most neurons in big
language models have one clear job and that interpretability may soon be solved. ARC's Jacob Hilton replied that
naming what neurons do is not the same as understanding a network, and pointed to a 432-number network that ARC
has not fully understood. ARC's challenge: *calculate* that network's accuracy from how it works, at least as
precisely as testing it on random examples does for the same effort.

**1. The essay: half right.**
- Neurons are sparser and somewhat easier to explain than random directions.
- Its "90% of neurons have one job" style of number is a weak measure: our blinded agents also called many decoy
  directions "one job".
- Our pre-registered test, after independent review, is *inconclusive*: the test was too small to decide.

**2. How the 432-number network works: about 80% understood.**
- Parts: a "highest so far" tracker, a "new record" alarm, checkers comparing the newest number with the high score,
  and rotating "clocks" whose angle records how long ago the biggest number arrived (measured 42.7 degrees per step,
  predicted 42.8).
- Its mistakes come from two situations: near-ties among the top numbers, and inputs that are almost all negative.
- Some prettier claims (a neat "harmonic" clock series, "all 16 neurons confirmed") were weakened or withdrawn by our
  own checks.

**3. ARC's challenge: not solved.**
- About 30 different attempts (literature-inspired, cross-disciplinary, "think like Feynman/Tao/Laplace/Poincare"),
  all scored independently on one leaderboard.
- **Every pure calculation loses** to plain testing on the 10-step network, by 8x to over 300x.
- **Best working method ("srb", smart testing):** about 1.6-2.4x more efficient than plain testing, and the only one
  that wins on every group of test networks, including independently trained ones. It still tests examples, so it does
  not count as ARC's kind of solution.
- **New map of where calculation works:** on short sequences (3-4 numbers), pure calculation beats testing by up to
  ~500x. Around 5-8 numbers (depending on network width) it collapses, because mistakes stop being neat thin strips
  and become rare, wide, irregular patches.

**ARC's public position (found 2026-10-17, see [lit/web_arc](lit/web_arc/README.md)):** on 2026-08-17 Jacob Hilton wrote that
their progress on AlgZoo is "fairly marginal and we are not much closer to succeeding at the challenge". ARC's only
published estimators (for a 32-parameter model) were, by our reproduction (E32), 2.4x (B) and ~10x (A) worse than
sampling in MSE at ARC's own operation counts, which ARC called "roughly match" (A is ~1.4x when operations are counted
as implemented, as our leaderboard does). So the private claim that ARC has cracked M_16,10 is doubtful on public
evidence (H-E5), and ARC's demonstrated bar is an MSE ratio of roughly 2-10, not our 1.

**First positive (2026-10-17, E36, independently confirmed by R2):** for the 32-parameter M_4,3, an accuracy estimate
derived from the network's mechanism (no sampling, no region listing, nothing fitted) beats random sampling ~50x in
squared error under ARC's protocol, where ARC's own estimators were 2.4-10x worse. Scope: one training seed; it does not
yet transfer to the other four (T75), and the ladder to bigger models is T73.

**Correction (E46):** the (8,4) loss in E45 was a bug in the 4-D probability calculation; with it fixed the recipe
**passes (8,4)** on all 5 seeds and noisy copies (MSE ratio <= 0.27). At (8,5) it is still accurate (error 0.0002 pp, seed 0) but costs 2.2e11 operations; the cost
grows ~250-5000x per extra input, so **M_16,10 projects to >=1e17 operations (E48): the challenge remains unsolved**. Earlier text: the recipe at (8,4) appeared to lose to sampling (MSE ratio 3-48) and its cells grow faster than the number of
linear regions (cells/regions 2.8, 5.4, 33 for T=3,4,5). Projected cost for M_16,10 is >=1e15 operations, so it was not
run. **The challenge remains unsolved**; the route needs a new idea that avoids following every gate switch (see
[DRI_HANDBOOK.md](DRI_HANDBOOK.md) section 6).

**4. The question to ask ARC.** If they have solved the 10-step network, they must have a compact way to handle
those rare wide patches. That is exactly where every approach here broke.

## Technical summary

| Question | Verdict | Where |
|---|---|---|
| Q1 Essay's "one main job" claim (Qwen3-0.6B) | H1 61% A (borderline); H2 underpowered: **inconclusive** (pre-registered rule mechanically says "refuted"); H3 passes but is inflated by token overlap; LLM-free one-job proxy shows no neuron/random difference | [`RESULTS.md`](RESULTS.md), ledger H-Q1..Q4 |
| Q2 Legible vs understood (AlgZoo) | M_4,3 control passes; M_16,10: 14/16 neurons legible, 5-constant labels rebuild 54% (driven by the constant limit) | [`RESULTS.md`](RESULTS.md), H-A1/A2 |
| Q3 Mechanism of M_16,10 | running max / record detector / comparators / damped rotating modes (phase = time since max); error anatomy exact; Fourier/harmonic reading weakened | [`m16/FINDINGS.md`](m16/FINDINGS.md), H-M1..M8 |
| Q4 ARC challenge | **open**; best verified estimator srb (sampling) | [`INDEX.md`](INDEX.md) section 4b |

**Leaderboard v3 (final, 2026-10-17; supersedes the v2 table below):** compute charged as implemented, results on models an
estimator was tuned on excluded, 30 test models up to harsh noise, 32M-sample truths. **Every mechanistic (no-sampling) estimator
loses on M_16,10, by 37x to >4000x.** Fixed-node quadrature (sci_tao, sci_feynman) beats sampling on the seeds and mild noise
(MSE ratio 0.07-0.46) but not at harsh noise; srb no longer beats sampling. Summary: [results/leaderboard_v3_summary.txt](results/leaderboard_v3_summary.txt).

**Leaderboard v2** ([`m16/leaderboard.py`](m16/leaderboard.py); 18 models: original, 5 trained seeds, 12 noisy copies; 24 repeats; 95% CIs).
MSE relative to plain sampling at 64k model evaluations (below 1 is better):

| Method | Original | Seeds 0-4 | sigma 0.0025 | sigma 0.005 | Type |
|---|---|---|---|---|---|
| srb (`control_variate/`) | **0.42** | **0.45** | **0.50** | **0.61** | sampling |
| joint (`tie_lines/`) | 0.34 | 0.72 [0.48,1.01] | 0.51 | 0.53 | sampling |
| boundary | 0.61 | 0.68 | 0.52 | 0.68 | sampling |
| stratified | 0.62 | 0.61 | 0.54 | 0.72 | sampling |
| tube | 0.14 (single value) | **1.23** | 0.46 | 0.75 | deterministic quadrature |
| plain sampling (calibration) | 1.06 | 1.15 | 1.13 | 0.90 | - |

**Deductive attempts (all fail vs sampling):** probe v2-v4, observer (E21), propagation (E13), cone enumeration (E14),
rank orderings (E10), FORM/SORM (E17, best bias +0.15 pp), branch-and-bound (E19, certified nothing),
Laplace/Feynman/Tao/Poincare scientists (E24-E27).

**Key facts established** (ledger IDs):
- H-M7: 1e7+ linear regions (Chao1 lower bound 3.8M, still rising).
- H-E7: gate pieces (~1.4 deg) are much finer than the correct/wrong pattern (5-20 deg), so exact path integration wastes compute.
- H-E8: even an oracle additive description of the tie band misses by 4%.
- H-E9 (refuted as root cause): margin cancellation is real but does not track failure.
- H-E10: deterministic summaries of the low inputs always overestimate accuracy (Jensen-type, 60/60).
- H-E11: calculation beats sampling up to T~4-5, collapsing as the band integrand turns heavy-tailed.
- H-E12: the rare wide multi-cell bands are the crux.

**Corrections made along the way** (from the independent review R1 and later checks): Qwen verdict softened to
inconclusive; random-direction control found to be neuron-dominated; one-job proxy null result added; stratified gains
corrected to 2.1/1.8/1.4x (and lower with generation cost); "harmonic series" weakened; "all 16 neurons confirmed"
corrected to 13 in groups; tube's high-budget gain not confirmed (bias floor); orig ground truth fixed by pooling 28M
samples; a "62x" efficiency figure retracted as an arithmetic error.

**Independence:** until 2026-10-17 no ARC AlgZoo material published after January 2026 was read. On 2026-10-17 the project owner relaxed this rule (ARC's M_16,10 solution is reported to be unpublished); every later ARC source read is logged in [lit/web_log.md](lit/web_log.md). The claim that ARC has solved
M_16,10 (relayed privately) is unverified. [`m16/arc_eval.py`](m16/arc_eval.py) and [`m16/leaderboard.py`](m16/leaderboard.py) can score any estimator with the
same interface for a direct comparison.

**Next steps** (prioritized and claimable in [TASKS.md](TASKS.md)): [`INDEX.md`](INDEX.md) section 5 (top: a stochastic reduction tracking the exact law of the tie margin; explaining
the T~5 collapse; an optional brute-force exact reference with more compute).
