# T190 exact time-shift segment reuse

## Question
What can shared recurrent weights actually reuse across time, and which
constraints still prevent reusing whole correctness subtrees?

## Method
Claim25723b1d precedes implementation. For each fixed gate word, construct
exact affine maps and preactivation planes in incoming hidden state plus
local segment inputs. Absolute time enters only through coordinate insertion.
First five H2 synthetic models, every word of lengths1..3, exact direct
masked/ReLU controls, adjacent-segment composition and shifted coordinates.
No trained capture or geometry/mass enumeration in this phase.

## Status
Initial synthetic phase complete at484b9599 and independently verified0132cd8f (checker22daa2a8),1.394151262reviewCPU. Next saved-only prefix-shared cost census claimed79c49477, sourcef601aeb1, execution gate pending. No trained census executed.

## Results
All420 fixed words pass:5220 masked and5220 direct ReLU checks,10440 shifted-coordinate checks and720 adjacent compositions. Both incoming-domain/truth counterexamples pass. The31 coefficient-zero ON words are sufficient infeasibility flags, not a complete feasibility census;165 compatible fixed points are not probability coverage. PROOF.md separates algebraic reuse from geometry, remaining depth, readout, truth rank and integration obligations.

Recorded command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python review/time_shift_reuse/controls.py` with external `ulimit -v 2500000`; internal19/20CPU. Immutable output guards prevent accidental overwrite; do not rerun completed controls.

## Compute accounting
Completed controls1.669981880CPU,36952KiB peak RSS; exact reduced coefficient bit maxima4/4. One thread2GB, review19/20CPU. Exact template arithmetic counted separately;
verification, conversion, hashing/control/storage and big-integer costs are
not free and remain separately recorded or unpriced. No Monte Carlo inputs,
trained model calls, quadrature or speed/cost advantage inferred.

## Claims
[inferred, independent source/theory and saved checks pass] Fixed gate-word templates are invariant
under time-coordinate insertion for identical recurrent weights.
[speculative] Their reuse may reduce a meaningful portion of estimator cost.

## Alternatives not ruled out
Weight sharing might help only arithmetic, while distinct cone/rank domains
still require all geometry: synthetic incoming-state/rank counterexamples
distinguish these. Apparent repeated words might arise from duplicate initial
charts: any later history census must deduplicate histories first. A template
cache can cost more than incremental construction; no speed claim from hits.
Broader affine symmetries/domain-restricted reuse are not excluded by misses.

## Failure modes
Incorrect gate-zero ownership, omitted incoming-state variables, mixed weights,
coordinate/rank relabeling errors, infeasible word counting, duplicate histories,
unpriced exact/cache setup, or equating conditional map equality with probability
or correctness equality. All counterexamples and failed attempts are preserved.
