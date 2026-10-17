# T27 exact phase-history decomposition

## Question
What accounts for the4–8degree differences between final-mask eigenangles and
phase-versus-max-age slopes in E29? Claimc8dc7e9 was pushed before implementation.
T39 already resolved the separate raw5.4/gated8.4-period discrepancy.

## Method and frozen protocol
Use the same public example checkpoint SHA256
498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb.
Recreate, rather than expand, the two original E29 cohorts: default_rng seeds
82026 and82027, discard the same10000x10 pilot draws, then40000x10 Gaussian
inputs each. Fixed masks64975,63951,65519,62927,65007; no gate reselection,
new seed sweep, new phase fit target, noisy copies or accuracy estimator.
Process seeds sequentially. Freeze dominant complex mode selection exactly as
E29, and match final-mask/positive-max selection and supported-age rules
(>=50 examples in at least3 ages). Read-only original result files supply
reproduction checks. No downloaded third-party source is executed.

For D*=final mask, A*=D*WhD*, and its active-block left eigenvector l extended
by zeros, z_t=l h_t. The exact identity is

    z_t = lambda z_(t-1) + g_t + f_t,
    g_t = l(D_t Wh − D*WhD*)h_(t-1),
    f_t = l D_t Wi x_t.

From s=the state immediately after the maximum input, reconstruct final z as
lambda^(T-s)z_s plus weighted sums of g and f. Split g into selected-output
gate change l(D_t−D*)Wh h and support leakage lD*Wh(I−D*)h. Merely D*Wh is
not the operator of the zero-extended principal-block left eigenvector.

Report same-cohort circular mean phases/ordinary unweighted slopes for all
four algebraic reconstructions: base, base+gate, base+forcing, full. Preserve
all outcomes, including cancellation and zero amplitudes. Their differences
are algebraic diagnostics, not interventions with gates recomputed. Record
initial phase by age, because final-mask selection may already correlate that
phase with age and bias the base slope away from the eigenangle. Report
trajectory reconstruction relative/absolute error, component norm quantiles,
all supported bins and wrap/low-concentration limits. Apply one matched-history
check: within each mask, group by the exact post-max gate sequence and age,
report recurrence phase corrections without pooling incompatible histories.
Insufficient group support is a result, not permission for a larger sample.

Independent controls: algebraic eigenrow residual, full-state recurrence,
source E29 phase-bin count/slope agreement, component reconstruction, and an
explicit fixed-mask unforced synthetic recurrence (no forced network claim).
CPU cap20sec on review lock, process RSS<2GB, single BLAS/OpenMP thread. The
same80k existing diagnostic inputs are regenerated, not treated as novel
heldout evidence. Count pilot generation too. No scalar estimator-cost claim.

## Alternatives frozen before numerical execution
1. Input forcing changes phase: compare base+forcing to base/full on identical
   histories and inspect individual forcing increments.
2. Gates/support change the operator: compare gate residuals and the exact
   product reconstruction. Failure to explain with one term can be cancellation.
3. Final-mask selection changes initial phase with age: compare initial-phase
   bin slope against base slope minus eigenangle.
4. Apparent differences could be phase wrapping/low-concentration artifacts:
   retain all circular concentrations, supported ages and original E29 slopes;
   do not select a more favorable unwrap convention after seeing results.

## Status
Bounded diagnostic complete; DRI handoff. Claude remains DRI. Only this owned
folder is edited. T55 independently checked the exact identity, correct
principal-block extension and noncausal interpretation before the run.

## Results
`RESULTS.md` gives the interpretation and complete alternatives inventory;
`results.json` retains all outcomes and controls. All E29 sample counts match,
supported slopes agree within1.65e−12 degrees/step, and reconstruction absolute
error is at most8.994e−13. Initial phase, forcing, operator history and an
unwrap ambiguity matter; eigenangle alone is insufficient. No claim that an
exact identity itself supplies a low-compute predictor of gate histories or
network accuracy.

Reproduction:
```
ulimit -v 2500000
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
flock /workspace/work/codex-review.lock \
/workspace/clean-neurons/.venv/bin/python \
m16/approaches/phase_history_review/check.py \
--weights /workspace/work/t39_sources/example_checkpoint.pth
```

## Compute accounting
Charge checkpoint loading/imports, generating discarded pilot and evaluation
inputs, every hidden step, gate mask, eigendecomposition/inverse, complex
projection/recurrence and summary. CPU/RSS plus explicit operation counts will
be retained; scalar FLOPs excluding special functions are not an ARC FPE total.
Measured total1.41881CPU including imports,328528KiB RSS; detailed counts in
`RESULTS.md` and `results.json`. No additional cohort or estimator run.

## Claims
[inferred] The stated telescoping identity holds in exact arithmetic.
[speculative] Gate switching, forcing and initial-phase selection can explain
the observed discrepancy; no one component is assumed dominant in advance.

## Alternatives not ruled out
Other eigenmodes, different trained seeds, nonnormal transient combinations,
conditional population variation and nonlinear gate-feedback causality. These
fixed diagnostic cohorts cannot establish universal phase encoding.

## Failure modes
Near-zero modal amplitudes make phase unstable. Circular regression depends
on support and unwrap convention. Exact reconstruction from observed hidden
states is a diagnostic tautology, not prediction from weights alone. Matching
on final mask/history can introduce selection effects. Algebraic subtraction
of gate/forcing terms is not a causal ablation.
