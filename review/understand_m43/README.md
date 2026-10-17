# T74: independent check of E36 / H-U1 (mechanism-derived M_4,3 estimator)

↑ [TASKS](../../TASKS.md) · [INDEX](../../INDEX.md) · reviewed: [understand_m43](../../m16/approaches/understand_m43/README.md) · protocol and truth: [arc_m43_repro](../../m16/approaches/arc_m43_repro/README.md)

Reviewer: DRI sub-agent (Claude session 011hwZqHr8rxiHqJ7yTBUF1z), 2026-10-17. The reviewed folder was not edited.
All scripts are in this folder and import the reviewed code read-only.

## Question

Does the E36 claim hold up when checked independently? The claim is: tier B of `est_mech.py` uses no fitting, no full
region enumeration and no random inputs, and gets RMSE ~0.3% and an MSE ratio ~0.02 against plain sampling at equal
ops under ARC's protocol (absolute σ = 0.05, 256 copies per seed, 72 ops per sample). Held-out seeds 4-5 agree.

## Verdict (summary)

| item | verdict | key number |
|---|---|---|
| 1. Fresh noise seeds and independent truth | **confirmed** | seeds 10-13 (1024 new copies): RMSE 0.31%, MSE ratio **0.018 (95% CI 0.007-0.032)**. exact.py agrees with 1e7-sample MC on 29 models (rms z = 0.89, max \|z\| = 1.72) |
| 2. "Nothing fitted" | **confirmed** (no continuous constant comes from model outputs or truths). The discrete choices pose low risk on noise seeds and do not transfer across training seeds | eps changed 0.02→0.2: output identical. The less-tuned variants B− and C also beat sampling on fresh seeds (0.049, 0.029) |
| 3. Ops counting | **partly**: honest in structure, about 24% undercounted by a strict recount. Wall-clock time strongly favours sampling | strict/charged = 1.22-1.24. Ratio 0.018 → ~0.022 (or unchanged if sampling is also charged strictly). CPU: 20 ms/model vs 0.07 ms for 191 samples |
| 4. Robustness | **partly**: robust to σ and to accuracy-preserving perturbations. Breaks off the trained-seed-0 mechanism and in a narrow "slope ≈ 1" window | σ = 0.02: ratio 0.0002; σ = 0.1: 0.157. Worst case inside \|Δw\| ≤ 0.1 at constant accuracy: 0.06 pp. Unconstrained worst case: 6.9 pp. Noisy zoo seeds 1-4: B fails (0.96-3.8) and C fails on 2 of 4 |
| 5. Mechanistic / deductive | **confirmed** | no RNG is called (guarded run). All geometry is deterministic |

**Recommendation to the DRI.** Move H-U1 to **confirmed, with scope**: *on M_4,3 seed 0 under ARC's protocol
(absolute σ ≤ 0.1, ops counted with the m43.Ops convention), a deductive mechanism-derived estimator beats
plain sampling at equal ops (MSE ratio ≈ 0.02 at σ = 0.05).* Keep these caveats in the ledger. (a) The result does
not transfer to other training seeds: tier B fails on zoo seeds 1-4, and so does tier C on noisy copies of seeds 2-3.
(b) On a strict op count the ratio is ~0.022. At equal wall-clock time in this numpy implementation, sampling wins by
~5×. (c) It is no cheaper than exact enumeration (13.8k vs 17.5k ops). (d) There is a failure window where a
crossover τ slope sits just past 1 (errors up to −5 pp, in both B and C).

## Method and results

Environment: `source .venv/bin/activate`, `cd review/understand_m43`, every run with `OMP_NUM_THREADS=1` (1 core).
Total compute for this review is about 12 CPU-min.

### 1. Fresh seeds and independent truth: confirmed

*Independent truth* (`python truth_mc.py`, log `truth_mc.log`). I wrote my own forward pass and my own noise draws
(`common_review.py`). The weights are loaded from `alg_zoo` directly and checked: they equal `m43_weights.npz`
exactly, the network is a ReLU RNN with no biases, and my forward pass matches the torch model to 4e-4 in logits
with identical argmax on 1e5 inputs. My noise draws equal `m43.noisy_models` to the bit. I ran plain MC with **1e7
samples per model** on 29 models: base, zoo seeds 1-4, 12 copies at σ = 0.05 (seed 10), 6 at σ = 0.1 (seed 20) and
6 at σ = 0.02 (seed 30). exact.py agrees on all of them: mean z = −0.27, rms z = 0.89, max |z| = 1.72, with
s.e. ≈ 1e-4. So exact.py is a valid truth at the 1e-4 level, about 30× below the estimator's RMSE.

*Fresh noise seeds 10, 11, 12, 13* (256 copies each, σ = 0.05; `python fresh.py 0.05 10 11 12 13`, log
`fresh_0.05.log`). The mean truth is 0.882 and mean p(1−p) is 0.1019, close to the author's 0.1024.

| tier | RMSE | bias | max \|e\| | ops | MSE ratio [95% bootstrap CI] | author (seeds 0-3) |
|---|---|---|---|---|---|---|
| A | 1.15% | +0.79% | 9.4% | 4,545 | 0.082 [0.063, 0.103] | 0.083 |
| A2 | 0.78% | +0.23% | 7.8% | 8,367 | 0.070 [0.043, 0.100] | 0.066 |
| B− | 0.53% | +0.02% | 6.5% | 12,727 | 0.049 [0.024, 0.081] | 0.059 |
| **B (claim)** | **0.31%** | **+0.05%** | 4.6% | 13,765 | **0.018 [0.007, 0.032]** | 0.019 |
| C | 0.30% | +0.00% | 4.7% | 23,164 | 0.029 [0.010, 0.052] | 0.029 |

Tier B's RMSE per seed is 0.14%, 0.32%, 0.36% and 0.36%. The CI is wide because a few copies with a lost crossover
margin dominate the MSE (max error 4.6%). The upper end of the CI is still 30× below 1.

*Sampling baseline checked empirically, not only by formula.* I drew 20 independent 191-sample estimates per copy,
191 being tier B's mean ops / 72. The empirical RMSE was 2.32%, against 2.31% from the formula. At σ = 0.1 the two were
2.97% and 2.97%; at σ = 0.02, 1.63% and 1.62%.

### 2. Audit of "nothing fitted": confirmed (discrete-choice risk is low on noise seeds, high across training seeds)

I read `est_mech.py` line by line. Its only imports are `Ops`, `clip` and `frac` from `m43`. It does not touch
exact.py, results.json or any truth. In the code, the variable `truth` means the true class of a geometric
half-chamber, not a ground-truth accuracy.

| constant / choice | where | source | chosen using outputs or truths? |
|---|---|---|---|
| CIRCLES, half-chamber normals `n_true`, `n_side` | L38, L59-60 | model-free geometry (ordering chambers) | no |
| equidistance split `n_split` (median equidistant from its neighbours) | L61 | hand-chosen, model-free | no. I checked that the 24 halves partition the sphere: total mass 1.0000000000000004 |
| A2, t_k, strip bounds, A3 per strip | L223-232 | closed-form functions of the weights | no |
| `rmid` (midpoint of a strip, used to read g2) | L230 | gates are linear in r inside a strip, so any interior point gives the same gates | no (exact by construction) |
| ratio thresholds ν (upper envelope) | L75-87 | Wo | no |
| "dead → class 0" | L103, L161 | argmax tie rule of the runtime | no (a convention, not a fit) |
| anchor point (offset eps = 0.05) | L241-245 | hand-chosen | no. **Insensitive**: eps = 0.02, 0.1 and 0.2 give bit-identical estimates on all 1024 fresh copies and on 512 copies each at σ = 0.1 and σ = 0.02 |
| anchor-pair rule for x0 = x1 and x0 = x2, fallback when \|act\| ≠ 2 | L246-249 | discrete design choice made after looking at seed-0 mechanism and noise seeds 0-2 | discrete, informed by exact per-family residuals (diag_*.py) |
| resolve only c12, arc reading at the edge midpoint, gate-crossing test | L179-196, defaults | discrete design choice (same provenance) | same |
| completion with the nearest-to-switching neurons | L201-206 | discrete design choice | same |
| \|S\| = 1 readout = argmax Wo[:, S0]; \|S\| ≥ 3 frozen linear | L162-167 | follows from the network's own readout | no |
| tolerances 1e-12 and 1e-9 | various | numerical | no |

*Risk from discrete choices made while looking at seeds 0-2.* The author's held-out seeds 4-5 gave 0.016. My fresh
seeds 10-13 give 0.018. So on the noise model there is no sign of overfitting. The variants that use fewer of the
tuned choices also beat sampling clearly on fresh seeds: B− (no completion) 0.049 and C (no anchor, no
family-specific rule) 0.029. The risk does show across training seeds, where the anchor-pair design is wrong (item 4).
That limitation is already disclosed in E36.

### 3. Ops counting: partly confirmed

*Structure.* Every geometric primitive the estimator uses is charged through `m43.Ops`: clipping, solid angles, A2/A3
compositions, envelope divisions, active-set reads, gate-crossing tests and completion distances. The only items not
charged are model-independent: the 24 half-chamber polygons and their r-ranges, computed once at import. This mirrors
how exact.py treats its start cones. The same counter is used for exact.py's 17.5k, so the comparison with
enumeration is like-for-like.

*Independent strict recount* (`python ops_audit.py`). I wrapped `clip`, `frac` and `_read_point` with my own count
under stricter rules: every comparison or sign test costs 1, a new clip vertex costs 13 rather than 10, a
solid-angle triangle costs 23 rather than 21, the arc-point search (2 per vertex, not charged at all in the
original) is added, and completion gets an upper bound. Before wrapping, I asserted that the unwrapped charge equals
the reported charge.

| model | charged | strict | strict / charged |
|---|---|---|---|
| base | 14,724 | 18,238 | 1.239 |
| zoo 1 / zoo 2 | 12,729 / 12,903 | 15,572 / 16,001 | 1.223 / 1.240 |
| 6 noisy copies (σ = 0.05) | 11.7k-15.6k | 14.4k-19.2k | 1.232-1.242 |

The mean strict/charged is **1.236**. Charging tier B strictly but sampling at 72 ops gives an MSE ratio of
0.018 × 1.24 ≈ **0.022**. Charging sampling equally strictly (12 ReLU compares, 2 argmax compares and 3 compares for
the label, about 89 ops per sample) leaves it at ≈ 0.018. Either way the conclusion stands. The undercount mostly
comes from comparisons, which ARC's multiply-add convention does not charge.

*CPU time* (measured in `fresh.py`). Tier B takes **20 ms per model** on one core. Plain sampling at the same op count
(191 samples, vectorised numpy) takes **0.07 ms**, which is ~290× faster. At equal wall-clock time, sampling could
draw ~55k samples, for an RMSE of ~0.14% against B's 0.31%. That is a wall-clock MSE ratio of ~5 **in favour of
sampling**. This comes from Python and numpy overhead in a branch-heavy geometric code. A compiled implementation
might close most of the gap, but that is untested. ARC's protocol counts operations, not seconds, so the claim stands
under that protocol.

### 4. Robustness: partly confirmed

*Other noise levels* (`python fresh.py 0.1 20 21`, `python fresh.py 0.02 30 31`; 512 fresh copies each):

| σ (abs.) | mean acc | B RMSE | B max \|e\| | B MSE ratio [CI] | C ratio | A ratio |
|---|---|---|---|---|---|---|
| 0.02 | 0.946 | 0.03% | 0.08% | 0.0002 [0.0002, 0.0003] | 0.0002 | 0.0098 |
| 0.05 | 0.882 | 0.31% | 4.6% | 0.018 [0.007, 0.032] | 0.029 | 0.082 |
| 0.10 | 0.768 | 1.18% | 6.1% | 0.157 [0.118, 0.197] | 0.230 | 0.65 |

The estimator beats sampling at every level, but its advantage shrinks from ~5000× to ~6× as σ grows from 0.02 to 0.1.

*Zoo seeds 0-4, clean* (`python zoo_clean.py`): this reproduces the author's table exactly. Tier B errors are
−0.01, −7.12, −1.63, −1.49 and −7.36 pp. Tier C errors are −0.02, −0.56, −1.49, −0.94 and −0.97 pp.

*Noisy copies of zoo seeds 1-4* (new; σ = 0.05, 64 copies each, `adversarial.py` part 4). The ratios are approximate,
at a fixed 14k ops for B and 23k for C.

| zoo seed | mean acc | B RMSE (bias) | B ratio | C RMSE (bias) | C ratio |
|---|---|---|---|---|---|
| 1 | 0.721 | 6.2% (−6.0) | 3.8 | 0.94% (−0.45) | 0.14 |
| 2 | 0.734 | 4.0% (−2.9) | 1.6 | 2.6% (−1.8) | **1.07** |
| 3 | 0.785 | 4.1% (−2.7) | 2.0 | 3.7% (−2.2) | **2.6** |
| 4 | 0.571 | 3.4% (−2.3) | 0.96 | 1.6% (−0.41) | 0.36 |

This adds to E36's caveat: the generic tier C, too, **loses to sampling on noisy copies of zoo seeds 2 and 3**.

*Function-preserving reparametrisations* (part 1). I permuted the hidden neurons and applied random positive
per-neuron rescalings (factors e^U(−1,1)) to the base model and 20 noisy copies. Tier B changed by at most 1e-15, as
it should, since accuracy does not change. This test could have failed if the code depended on neuron index or scale.

*Breaking the mirror symmetry* (part 2b; noise only on the u-half neurons 0 and 1, i.e. their Wi entries, Wh rows and
columns, and Wo columns; 128 copies each). At σ = 0.05: B RMSE 0.11%, ratio 0.003. At σ = 0.1: 1.18%, ratio 0.17. At
σ = 0.2: 1.71%, ratio 0.26. The estimator does not rely on the symmetry, as claimed.

*Pushing a crossover τ slope across 1* (part 2a). I scanned a single weight over ±0.4 and computed β_k in chart +,
strip 0 in closed form.

- Wh[0,3] moves β_0 from 1.24 (partner β_2 = 0.81). B is accurate (|e| ≤ 0.05 pp) while β_0 ≥ 1.04. In the window
  **β_0 ≈ 0.89-0.99** (Δ = −0.35 to −0.25), B errs by **−1.2, −3.2 and −5.0 pp**, and C errs about the same. The error
  vanishes again at β_0 = 0.84. Accuracy in the window is 0.69-0.79.
- Wh[2,3] moves β_2 from 0.81. B is accurate up to β_2 = 1.02. At **β_2 = 1.06-1.15** it errs by **−4.8, −4.4 and
  −3.8 pp** (C the same). Accuracy there is 0.90-0.93, so the model is still good.
- Wh[1,1] (β_1 in strip 4, 0.81 → 1.21) causes no failure (≤ 0.3 pp).

This is the author's disclosed "lost crossover margin" failure mode. It needs a 5-7σ change to one weight, or a
rarer combination under σ = 0.05 noise (it produces the 4.6% max error). Both tiers fail there, so this is a limit of
the first-order frozen-gate approximation, not of a tuned choice.

*Worst case inside the box |Δw_i| ≤ 0.10 = 2σ on all 32 weights* (part 3; random-restart hill climbing on
|est_B − exact|, 6 restarts × 150 steps each).

- Unconstrained: the worst error found is **+6.9 pp**, at exact accuracy 0.635. Tier C errs by +6.8 pp on that model.
- Constrained to exact accuracy ≥ base − 1 pp (climbs started at the base model): the worst error found is **0.06 pp**.
  Perturbations that preserve accuracy did not break the estimator.

### 5. Mechanistic in ARC's sense: confirmed

`ops_audit.py` replaces numpy's RNG entry points (`default_rng`, `random`, `standard_normal`, ...) with functions that
raise, then runs tier B on 9 models. Nothing raised. The code has no sampling, no quasi-random grid and no use of
model outputs at random inputs. Everything is closed-form linear algebra plus exact solid angles of convex spherical
polygons. Runs are deterministic: repeated runs and different eps values give bit-identical outputs. There is no
memoisation across calls (the per-call `env_cache` is local), so it complies with the T70 rule.

## Compute accounting (this review)

Compute consists of exact.py truths for 2,560 noisy copies and 2,000 adversarial or scan models, the 1e7-sample MC on
29 models (2.9e8 forward passes), tiers A-C and the eps variants on 2,048 copies, and 20 × 191 sampling estimates per
copy. The total is about 12 CPU-min on one core, with at most 2 processes at once.

## Claims

- [measured] On 1024 never-used noise copies (seeds 10-13), tier B has RMSE 0.31% and MSE ratio 0.018
  [0.007, 0.032] against plain sampling at equal ops. The sampling RMSE was checked empirically.
- [measured] exact.py agrees with an independent 1e7-sample MC truth on 29 models (max |z| = 1.72).
- [measured] No constant in est_mech.py is derived from model outputs or truths. eps has no effect over 0.02-0.2.
- [measured] Strict op recount is 1.24× the charged count. Wall-clock time is ~290× that of sampling at equal ops in
  this implementation.
- [measured] At σ = 0.1 the ratio is 0.157. On noisy copies of zoo seeds 2-3 even tier C loses to sampling.
  Perturbations that preserve accuracy within |Δw| ≤ 0.1 found no error above 0.06 pp. Unconstrained ones found 6.9 pp.
- [inferred] The residual risk is concentrated in models where a crossover slope is near 1. This is a limit of the
  first-order (frozen-gate) approximation that both B and C share.

## Alternatives not ruled out / checks not done

- The hill climb is a local search with 6 restarts. Larger worst cases may exist, so the constrained 0.06 pp is a
  lower bound on the worst error, not a guarantee.
- The truth check covers 29 models at 1e-4 precision. The other 2,531 truths rely on exact.py, now validated on the
  sample.
- I did not reimplement the estimator, so a shared bug in `m43.clip` or `frac` would affect both the estimator and
  exact.py. But the MC truth, which does not use them, agrees with exact.py, so `clip` and `frac` are correct to 1e-4.
- I did not test a compiled implementation for wall-clock time, or relative (scaled) noise.

## Failure modes found

1. A crossover slope just past 1 (β ≈ 0.89-0.99 for the outer neuron, or 1.06-1.15 for the inner one): −4 to −5 pp,
   in both B and C.
2. Other training seeds: tier B fails (clean and noisy). Tier C fails on noisy seeds 2-3.
3. Wall-clock time is ~290× worse than sampling at equal ops.

## Files

`common_review.py` (independent forward pass, noise draws and MC), `truth_mc.py` with `truth_mc.log`/`.json`,
`fresh.py` with `fresh_<σ>.log` and `fresh_<σ>_<seeds>.json`, `ops_audit.py` with `ops_audit.log`, `adversarial.py` with
`adversarial.log`/`.json`, `zoo_clean.py` with `zoo_clean.log`.
