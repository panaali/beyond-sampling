# Position patches: matched effects and a reachable rescue

This bounded test does **not** establish selective causal mediation by n3/n13
or the fast modes, and does not refute those roles. Arbitrary partial patches
produce an apparent negative, but none is locally reachable. A stronger
reachable construction succeeds numerically while necessarily changing other
coordinates and mostly losing the original controls' readout matching.

## Frozen position-swap test

1014 pairs satisfy the predeclared prefix-position and final-mask filters in
the first4096 E29 examples. The donor exchanges the largest and second-largest
input positions before step8. The remaining two inputs are identical. Original
recipient accuracy is.951677; full donor accuracy.964497. Resuming from the
full donor h8 reproduces the donor final state exactly. The average donor-minus-
recipient target contrast changes by26.7005 (descriptive SE.5600).

| Partial patch | Successful10% norm matches | Partial-minus-control contrast change, mean±SE | Patches with negative hidden coordinates | Same-cell linear feasibility |
|---|---:|---:|---:|---:|
| n3/n13 donor coordinates |626/1014|−.58021±.14012|0/1014|0/1014|
| E29 fast-mode span |279/1014|−10.77412±1.96216|274/1014|0/1014|

Matching jointly uses hidden L2 perturbation and immediate Wo-perturbation L2
norm, with no target contrast or outcome in the selection score. Controls are
other coordinate pairs or a predeclared64 random subspaces per mask. Effects
are evaluated only after the same two-input suffix. Complete unmatched results,
pair IDs, chosen controls and outcomes are retained in `results.json`.

These SEs summarize variation across fixed input pairs, conditional on selected
subspaces/matches. They do not include uncertainty over trained networks or the
finite random-control library. No significance test or post-hoc claim is made.

The recipient gate-cell map has8 input columns. For every tested target and
control, its complement-projected map has numerical rank8. Thus no nonzero
direction in that recipient cell changes only the tested subspace. Desired
partial patches have relative projection residual at least.06286 for n3/n13
and.10136 for the fast span. Coordinate patches stay nonnegative but still fail
this reachability test. This is a numerical local-cell statement, not a global
unreachability certificate; another gate history could have a different image.

## Strongest tested rescue: actual input witnesses

The follow-up preserves the desired group coordinates while minimizing the
unavoidable displacement of complementary coordinates. If U spans range(A8),
the minimum-norm feasible displacement is

    d = U (Q^T U)^+ Q^T(h_donor−h_recipient).

When the coordinate equality is feasible, minimizing ||d|| minimizes its
complementary part because the group part is fixed. Solving A8 dx=d provides
an input direction. A common target/control alpha is set to90% of the smaller
safe bound from **all ten** original preactivation inequalities, with zero
forms explicitly handled. Direct network runs on the perturbed input prefixes
provide the actual witnesses; no partial hidden vector is assumed reachable.

| Quantity | n3/n13 versus matched coordinate control | Fast span versus matched random span |
|---|---:|---:|
| Originally matched pairs |626|279|
| Positive safe steps |615|279|
| Target group-coordinate equality failures |0|0|
| Control group-coordinate equality failures |11|0|
| Median common alpha |.03302|.005034|
| Median target complementary/group norm |.5930|.9881|
| Preserved full donor input order |0|0|
| Original10% matches surviving with positive step |306|7|
| Changed gate histories in either witness |0|0|
| Largest forward h8 reconstruction error |7.82e−14|1.14e−13|

All original outcomes, including the11 infeasible control constraints, remain
in the result. Those pairs have alpha0, rather than a fabricated intervention.
Numerically witnessed target equalities hold within1.22e−15 for n3/n13 and
1.32e−14 for fast spans. The failed control maximum relative error is.8902;
it is not hidden in a pooled successful-witness statistic.

For all original matched pairs (including explicit zero-step failures), the
reachable paired contrast differences are−.02224±.01046 and+.01317±.02147.
These are small mixed-coordinate local changes, not completed position swaps.
The rank order of the full donor input is never recovered. Fast-span norm
matching survives for only7 pairs, so no stable selective effect can be inferred
there. The initial partial-patch negative is not rescued as a causal negative.

## Independent checks and limitations

T55 independently checked the local affine-image criterion, the need for all-
time gate inequalities, and the distinction between closed-cell feasibility
and stable strict gates. Its active-ambient-dimension caveat is respected:
we compute actual complement ranks, rather than infer them from H=16 alone.
The minimum-norm rescue was submitted for independent algebra/source review;
forward witnesses supply a separate numerical construction check.

Numerical SVD/pseudoinverse tests are not exact rational certification. Full
donor witnesses are genuine reachable states, but they change multiple hidden
coordinates. Partial patches fail one cell; global witnesses across other
cells were not enumerated. Immediate readout norm matching does not equalize
the nonlinear suffix. Reachable rescues hold original gates fixed and are only
local sensitivities; they do not demonstrate that a group stores a unique
feature. No hidden neuron or mechanism family is retired.

Plausible alternatives left open: new reachable controls matched **after**
constrained reconstruction, multi-cell constrained paths, conditional donor
resampling, different swap types/times, and other trained models. These are
distinct formulations, not permission to tune thresholds on this result.

## Reproduction and costs

From the owned worktree, prefix each command with single-thread environment,
`ulimit -v 2500000`, and the review lock:

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python \
m16/approaches/position_patch_review/check.py \
--weights /workspace/work/t39_sources/example_checkpoint.pth

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python \
m16/approaches/position_patch_review/reachable.py \
--weights /workspace/work/t39_sources/example_checkpoint.pth \
--out m16/approaches/position_patch_review/reachable_results_detailed.json
```

Initial phase1.13193CPU, peak208772KiB. Reachable phase1.03280CPU, then a
1.09564CPU repeat added target/control-specific equality failure and positive-
match counts; both raw artifacts are preserved and every previous number is
unchanged. Total executed numerical work3.26037CPU including imports/loading
per process, peak222532KiB. The stronger detailed artifact is authoritative for
failure classification. JSON serialization follows each timing snapshot.

Initial counts:61,240 hidden steps,6,084 readout vectors,8,112 gate-map updates,
5 eigen/5 span decompositions,320 QR controls,4,056 least-squares and4,056
complement-SVD solves,185,562 control projections. Each rescue run adds38,380
hidden steps,1,810 readout vectors,10,140 derivative-map updates,5 eigen/5 span
decompositions,320 QR controls,905 reachability SVDs and3,620 least-squares
solves. All three runs charge140,960 model Gaussian variates and16,896 control
variates each. No scalar FPE or estimator-performance claim is made.
