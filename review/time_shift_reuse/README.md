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
Initial synthetic phase complete at484b9599 and independently verified0132cd8f (checker22daa2a8),1.394151262reviewCPU. Saved-only prefix-shared cost census7c612afb independentlyc8fff4dd complete after claim79c49477/source gate14417f17. No trained matrix/cache or geometry execution. Further local optimization paused under new DRI directive130a293d pending evidence export and the requested existingH8T4 tree discriminator.

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

## Saved cost discriminator

Existing3853 simplex-history records deduplicate to450 histories, with parent-prefix counts1,2,10,66,450. The fair dense named baseline uses6160 multiplications/additions. At fixedcuts1/2/3, warm counts2184/4312/6160 plus cold template counts10528/2240/0 yield12712/6552/6160. No strict cold saving under this specified layout. Eight analytic count controls and four invalid guards independently14417f17; full saved contraction independentlyc8fff4dd. ControlCPU.029662113; censusCPU.018243328; no matrix or model evaluation.

This does not exclude sparse products, reset/state equivalence, different horizons, multiple cuts, factorization or genuine subtree-domain reuse. Dense arithmetic is tiny beside geometry/mass work and projected counts do not establish actual implementation cost. Neither failed cold cuts nor warm reuse justify declaring time-shift reuse solved or dead.
