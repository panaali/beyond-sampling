# T101: the training-amount axis of the ladder, M_4,3 at n_seqs = 2^28 down to 2^20

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [T75 recipe (E37)](../understand_m43/README.md) · [fibre recipe (E38)](../understand_ladder/README.md) · [R4 review / cache fix](../../../review/understand_ladder/README.md) · [R3 review / strict ops](../../../review/understand_m43_t75/README.md)

## Question

At the solved rung M_4,3 = `alg_zoo.zoo_2nd_argmax(4, 3)`, do the two frozen mechanism-derived recipes keep beating
sampling as the model becomes **less trained**? The models are `zoo_2nd_argmax(4, 3, n_seqs=2**k, seed=s)` for
seeds 0-4. They are checkpoints of the same five training runs, so k = 30 is the fully trained model of E37. The
ladder protocol (TASKS.md) says: one step at a time, stop at the first failing rung, diagnose, fix, re-check earlier
rungs. The two recipes are:

- **T75**: `est75.RECIPE = dict(K=(4,4), rule="adaptive", eps=0.1)` (E37/H-U2). It resolves the gates that matter by
  impact inside 24 half-chambers and freezes the rest.
- **F2L3**: the fibre recipe F2 with look-ahead L = 3 (E38), with R4's `id()` cache fix.

The first training-amount rung, 2^29, was done by Codex in T90 (board note: passed all 5 seeds with an independent
T55 check; folder `m16/approaches/training_axis_ladder/`, commit 6d47ac7). **That folder and commit are not on this
branch (`ccr-6a61f25c-wn80ii`), so I could not read or reuse its exact panels.** Following the DRI's instruction I did
not redo 2^29. I used the E37 panel convention, described below, and report k = 30 as the harness check.

(Status, results and verdict: see below.)

## Method

**Models per rung k.** For each zoo seed s = 0-4 there is the clean model plus a frozen noisy panel. The panel uses
the ARC protocol: absolute σ = 0.05 on all 32 weights, noise seeds 20, 21, 22, 23 × 256 copies, which is 1024 copies
per seed. This is `understand_m43/zoo43.noisy_copies`, the E37 fresh-seed convention. **The same noise draws are added
at every k** (common random numbers), so rungs differ only in the base weights. Truth is exact enumeration
(`arc_m43_repro/exact.py`) for every model.

**Recipes and the copies they are scored on.**
- T75 is scored on all 1024 copies per seed. It runs from `est75_e37.py`, a pinned copy of `understand_m43/est75.py` at
  commit d406c17, the version that produced E37. Another lane edits `est75.py` during this work (T85/T112). The pinned
  copy gives bit-identical estimates and op counts to the live `RECIPE` on 140 models (k = 30/27/24/21 × 5 seeds × 7
  models).
- F2L3 and the linear-region counts are scored on the first 256 copies per seed at k = 30 and on the first 128 per seed
  for k ≤ 28 (noise seed 20). The smaller panel is because of CPU limits (about 0.4 s per model). It runs from
  `review/understand_ladder/fibre_strict.py` (R4's strict recount), with R4's one-line cache fix from `fibre_fix.py`
  patched in at import (`lt.py`), and LOOK = 3.

**Ops.**
- T75 uses est75's own strict charge, the T74/R3 convention. R3's automatic recount found it fair to within 1.04×.
- F2L3 uses R4's strict count, n + x.
- Sampling costs 72 ops per sample (ARC). MSE ratio = MSE / (mean p(1−p) / (ops/72)).

**Pass rule (ladder protocol, E37/E38 aggregate convention).** A recipe passes rung k if, on every zoo seed, both
conditions hold:
- (a) the panel MSE ratio is < 1;
- (b) the clean model's single-model ratio is < 1.

A bootstrap 95% CI over copies is reported for each ratio, along with the share of individual copies whose own ratio
is ≥ 1. That share is reported for information only: a single copy can exceed 1 while the recipe still beats sampling
on average.

**Mechanism tracking** (`mech.py`, all k, clean models). Exact enumeration is used here only to describe the models;
no estimator uses it. Quantities tracked:
- the step-2 switch ratios r_k = x1/x0 at which each neuron switches, per chart sign(x0);
- which neurons store x0 at step 1;
- the error mass broken down by the number of active step-3 neurons, including the dead (all-off) share;
- the share of error touching a true boundary;
- the number of step-3 gates cutting a step-2 piece.

**Diagnostics** (development data only: noise seed 0, never the 20-23 panel).
- `diag_t75.py` attributes T75's error to the step-2 vs step-3 freezes.
- `diag_neuron.py` attributes the step-3 freeze error to individual neurons.

Files:
- [lt.py](lt.py): loaders, panel, recipe wrappers.
- [run_rung.py](run_rung.py): scores one rung and writes `rung_k{k}.json`.
- [verdict.py](verdict.py): pass/fail.
- [ladder.sh](ladder.sh) / [ladder2.sh](ladder2.sh): driver (stop at first failure) and `ladder.log`.
- [report.py](report.py): tables.
- [mech.py](mech.py): `mech_k*.json`, `mech.log`.
- [clean_scan.py](clean_scan.py): `clean_scan.json/.log`, a clean-model preview over all k.
- [diag_t75.py](diag_t75.py), [diag_neuron.py](diag_neuron.py).
- [est75_e37.py](est75_e37.py): the pinned T75.
