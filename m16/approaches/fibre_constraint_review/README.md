# T99 approximate confirmation-order constraints

## Question
Can dropping gate-versus-gate ordering constraints on already-correct fibre
pieces reduce trace work without spoiling the passed T3 recipe? Claim4ab9117
was pushed before this implementation. Claude retains all shared ladder code
and final recipe adoption. This is a local diagnostic, not a new passed rung.

## Method and frozen first cycle
Pin fibre_fast.py, fibre.py and fiber3.py from shared commit7888dc3. Preserve
unmodified source snapshots here for reproducibility; do not edit the author's
folder. Original F2L2fast, LOOK2, is the control. The reference already makes
an attached-error/finite-lookahead approximation; it is not exact integration.

Change ONLY the forward trace's nearest-next-gate versus other-ahead-gates
constraint call. Defer this call until the piece's output intervals have been
classified. If every interval is correct, omit this one call; otherwise keep
it. Keep all entry, clamp, class-margin and reverse-walk constraints. No
changing gate dynamics, first-index argmax, LOOK, sampling or fitted thresholds.
The local implementation records dropped calls/pairs and charges its decisions.
An exact deferred-but-retained mode must agree with original before testing
omission, since operation reordering itself should preserve the recipe.

Frozen strongest cheap rescue: same omission, but when any order constraint
was omitted in a trace, cap its angular validity width at pi/64, centered on
the traced angle and intersected with remaining validity/segment bounds.
This is a predeclared accuracy/cost tradeoff, NOT an error enclosure. All three
outputs (reference, omission, angular cap) are reported; no favorable choice.

First numerical panel: public H4T3 seed0 and the FIRST sigma.02 relative-RMS
noise copy generated from that SAME seed0 by NumPy default_rng(0), each matrix
in Wi,Wh,Wo order. Never select best seed. Separate exact3.py reference on
these fixed models; charge reference separately as validation. Also zero
readout, duplicate readout, and Wh0 synthetic controls with analytic1/3
accuracy when output is a constant first-index label. Check output rescaling
1e-8/1e8 on trained seed0 as a diagnostic of inherited tolerance sensitivity,
not as license to repair source silently.

One<=100CPU primary-lock cycle,2GB, one thread. Freeze no new Monte Carlo or
training. Existing model/source setup, proxy operations, CPU, trace/constraint
counts, dropped calls/pairs, range violations and failures are retained. Broad
H4/H5/H6 seed/noise panel follows ONLY after these controls and review; all
rungs must be re-passed before DRI adoption. No blind degree or threshold sweep.

## Alternatives and discriminators
1. Bookkeeping reordering itself changes the result: deferred-but-retained
   mode compared with original is an exact implementation control.
2. Omitted constraints expose hidden wrong intervals: compare exact geometric
   truth and frozen angular cap, preserving any lost accuracy or cost gain.
3. Tolerance/scale or positive-mass ties explain differences: constant and
   scaling controls run before inference about approximation quality.
4. Gate-order constraints are not the dominant cost here: charge construction,
   remaining class constraints and trace work, not just counts removed.

## Status
Protocol frozen; implementation not yet run. No result or pass claim.

## Results
Pending bounded first cycle.

## Compute accounting
Source setup and checkpoint acquisition logged. Scalar arithmetic counters are
inherited proxies (multiply-add convention); special functions, allocations,
Python overhead and model-independent family construction are not a certified
instruction count. Report CPU including imports separately and add local
predicate/cap charges. Validation truth and controls remain separate costs.

## Claims
[speculative] Confirmation-piece ordering may cause unnecessary refinements
when classification is insensitive to the changed gate sequence.

## Alternatives not ruled out
Impact-aware freezing, a rigorous readout error certificate, adaptive omission
from an explicit error budget, and higher-dimensional fibre formulations.

## Failure modes
Wrong interval topology after omission, inherited fixed tolerances/caps,
false confidence from one seed, proxy undercounting, and an angular cap whose
accuracy improves only by spending more than the original recipe.

## First cycle results
[measured] All3 constant-label controls pass to1e-10 in all four modes.
Deferred-but-retained order constraints reproduce the unmodified fast recipe
exactly on both trained models and all controls. The limited omission also
preserves the two trained estimates at printed precision while lowering proxy
cost by3.81%/4.15%. This is a scoped small gain, not the suggested62% overall
reduction: only one specific ordering call was removed on correct pieces.

| Model | Numerical truth | Reference estimate | Omission estimate | Reference / omission proxy ops | Capped proxy ops |
| --- | ---: | ---: | ---: | ---: | ---: |
| H4T3 seed0 | .9845503813585573 | .9845554916788959 | .9845554916788959 | 211218 / 203164 | 439885 |
| Same seed, first relative.02 copy | .8593001113320059 | .8595162512889670 | .8595162512889670 | 214688 / 205775 | 416801 |

The angular-cap rescue retains the same estimates but more than doubles
reference traces (1283/1266 versus445/475) and raises cost. It is not selected
as the default. Omission drops306/285 calls (702/497 candidate pairs), reducing
traces to434/455 and retained constraints to8746/8389. Deferred mode costs an
extra970/998 local decisions, which are charged. All methods' output scaling
checks at1e-8/1e8 agree within1e-12; full artifact preserves exact differences.

Whole phase3.355113876CPU including imports/setup,202376KiB RSS, no random
model inputs. Reference polygon accuracy on the original matches the pinned
ladder artifact. The source m43.py eagerly reads its baseline weight NPZ;
initial import failed because this dependency was missing from our snapshot.
No numerical model test ran in that attempt; its CPU was not measured. The
unchanged dependency was copied and hashed, while all evaluations still take
the explicit frozen models. This setup overhead is not silently treated as0.

Reproduce from the research worktree after downloading the checkpoint URL in
source_manifest.json to /workspace/work/t99_checkpoints/H4_T3_seed0.pth:

```sh
ulimit -v 2500000
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-compute.lock /workspace/clean-neurons/.venv/bin/python m16/approaches/fibre_constraint_review/pilot.py
```

[inferred] The small initial gain justifies a fixed broader discriminator;
it does not establish safe omission of other ordering calls. Full H4/H5/H6
seed/noise panel and independent review remain required. The constant cases
are correctness controls, not representative cost measurements. Proxy MSE
ratios in pilot.json omit uncharged setup/special-function details and are
not portable leaderboard results or rung passes.
