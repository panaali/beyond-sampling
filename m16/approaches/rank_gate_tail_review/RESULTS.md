# Rank support and frozen gates both matter

The fixed DEV experiment isolates two separate approximation errors. Replacing
the Gaussian closure by the exact rank-conditional input distribution changes
the frozen-map answer materially, but does not fix it. Recomputing the final
two gates is also insufficient at the predeclared0.1pp diagnostic tolerance.
This is a conditional Monte Carlo diagnostic, not a deductive estimator or a
benchmark against equal-cost sampling.

| DEV noise | Rank law, frozen map | Gaussian law, frozen map | Rank law, actual RNN | Rank law, final-two-gate rescue |
|---|---:|---:|---:|---:|
|.0025|97.2049%|92.1311%|94.1189%|92.7561%|
|.02|73.2335%|70.3125%|62.6259%|68.0512%|

Each entry averages all180 equal-probability position/second-value cells with
128 independent rows each. Rank-law/full-network uncertainty is.1527/.2358pp
SE. Gaussian entries retain the prescribed chamber label even outside the
chamber, so they are moment-closure functionals, **not accuracy under ordinary
Gaussian second-argmax labels**. Both models use the same fixed DEV noise seed1,
copy0 and per-matrix RMS recipe as T37; exact model hashes are retained.

## Separating the errors

The moment-matched Gaussian sample violates the prescribed top-two chamber on
42.7648% of rows; its actual second-largest position differs from the chamber
label on39.9479%. Mean/covariance matching cannot preserve rank support.

Within the frozen map, restoring the rank law increases the predicted accuracy
by5.0738±.2047pp SE at sigma.0025 and2.9210±.3024pp at.02. The resulting
rank-respecting frozen map still has bias **+3.0859±.1487pp** and
**+10.6076±.2294pp** against the actual network on the same rank inputs.
These biases exceed3SE plus.1pp. Removing rank leakage alone therefore does
not rescue this fixed gate approximation on either DEV model.

The predeclared exact-last-two-step rescue leaves biases
**−1.3628±.1767pp** and **+5.4253±.2394pp**, also beyond3SE plus.1pp.
It changes the error sign at lower noise and reduces the high-noise bias, but
does not meet the tolerance. Its frozen step8 hidden state has at least one
negative coordinate on23.2292%/28.8411% of rank inputs. The terminal correction
starts from that approximate state; it is not an exact two-input conditional
integrator applied to the true prefix law.

The full RNN under the Gaussian **fixed-chamber pseudo-label** scores59.9219%
and45.1693%. Combined with the other cells, the distribution×nonlinearity
interaction is29.1233±.3540pp and14.5356±.3777pp. It would be misleading
to assign this large interaction to one causal component or call those scores
ordinary network accuracy. Changing the distribution changes gate histories
and whether the prescribed label remains the true target.

## Marginals versus overlap

The independent map construction reproduces E30's analytic Gaussian summed-tail
estimates89.58113%/60.96725% and union ceilings93.41211%/73.83922% within
1e−11. The sampled joint Gaussian/frozen scores92.1311%/70.3125% lie within
these intervals, as expected. Exact overlap can help that approximation, but
does not change its marginal laws. The existing E30 oracle result remains a
separate finding on its original/test model groups, not a conclusion inferred
from these DEV values.

Average absolute per-competitor marginal changes are1.143/1.660pp when replacing
Gaussian inputs by rank inputs under the frozen map, and.904/4.484pp when
replacing the frozen map by the actual network on rank inputs. These are
finite-sample diagnostic absolute differences, upward biased by estimation
noise; the signed aggregate changes and complete cell marginal arrays are
retained. Do not interpret them as independently estimated population L1 norms.

## Controls and independent reasoning

Every rank sample has the specified largest/second positions and lies in its
declared Beta-CDF stratum. Both covariance matrices are positive definite
(minimum eigenvalues.04589/.05483). Empirical moment deviations are recorded,
not used to fit or modify the moment table. The table comes from separate
deterministic one-dimensional numerical integration, not these128-row cells;
its construction CPU is charged. Maximum mean-map logit discrepancies are
4.20e−13/2.71e−13. Constant zero readout with first-index argmax gives exactly
1/10 under balanced chamber labels.

T55 independently checked the exact conditional distribution: Phi(second) has
Beta(9,2) law; within equal-probability strata one must uniformly sample its
Beta-CDF coordinate and invert; conditional upper/lower coordinates are
independent truncated uniforms transformed by Phi-inverse. T55 also checked
the fixed-cell SE formula and the pseudo-label/interaction interpretation.
An independent high-noise numerical check using ordinary iid inputs assigned
to chambers, rather than this conditional sampler, was requested separately.

## Scope and alternatives

This rejects only the two fixed mean-gate maps and their tested terminal rescue
at the stated DEV diagnostic tolerance. It does not reject conditional
non-Gaussian integration, earlier-gate mixtures, exact support propagation,
other chambers or models, or gate-aware conditional estimators. No settings
were tuned against leaderboard test models or scored truths. Numerical inverse
CDF/moment errors are not certified, and descriptive3SE intervals are not
simultaneous exact coverage statements.

The next distinct formulation would retain input-rank constraints **and**
resolve selected earlier gates or a conditional mixture of reachable prefix
states. Repeating only overlap corrections, increasing Gaussian strata without
a gate model, or treating the final-two-step correction as solved would not
address the diagnosed errors. No new engine is implemented or claimed here.

## Reproduction and cost

```
ulimit -v 2500000
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python \
m16/approaches/rank_gate_tail_review/check.py \
--weights /workspace/work/t39_sources/example_checkpoint.pth
```

Total1.56614CPU including imports/loading/model-free table/setup, peak293640KiB.
Input laws each have23040 rows, reused across two models. Counts:230400 uniform
and230400 normal variates,23040 Beta inverse and230400 normal inverse arguments,
2 covariance factors,864 noise variates,921600 actual hidden steps,92160 rescued
steps,7200 mean/Jacobian updates including pinned E30 reproduction,92160 linear
and92160 actual logit vectors,46080 rescued logits,7200 analytic Gaussian tails.
Summaries/margins/cell assignment are included in CPU, with explicit arrays
retained. JSON serialization follows the timing snapshot. No portable scalar
operation bound or estimator MSE ratio is asserted.
