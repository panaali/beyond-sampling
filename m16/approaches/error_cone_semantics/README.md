# T139: independent output-error cone semantics

## Question
Do the fixed captured error simplices contain deterministic interior witnesses whose second-argmax truth matches the captured parent and whose independent direct network prediction is wrong? TASKS-only claim8d14e5b published before implementation. T131 owns geometry and mass replay; this is an independent semantic point check.

## Method
Freeze capture from recover_910 commitbad6890: error_capture.json SHA7cdd206f29c35dd5b46319b94d5ed5f5609e99873e9f0a6c73efe68fe9f48057 and error_captured.npz SHA15f9e5e483c69325c10fcbe4eaaa80f91503b177070ccec1ff2569a11c9436b6. Public H8T4seed0 checkpoint SHAb771f40fcae826e010b2e89fb698e2ec35026119bc9f66d3ff046dba537b07da.35098 simplices,96 parents. No source engine imports or geometry/mass changes.

At every simplex use the sum of all4 stored rays (coefficients exactly1, no model-adaptive witness). Independently evaluate h0=0, h_next=max(0,Wh*h+Wi*x_t), logits=Wo*h, first-index prediction. Stable descending input order gives second-argmax truth; tied input coordinates are separately reported. Compare both labels to captured parent. Store all logits, truths, predictions, input values and margins. Margin is logit_truth minus maximum nontruth logit; sort by absolute margin with original simplex index as tie-breaker. Select64 cases: all floating semantic failures first in index order (up to64), then smallest-margin remaining cases. Preserve and explicitly report if failures exceed the exact-check cap.

For selected cases convert every stored binary ray and checkpoint coefficient exactly to Fraction, sum rays exactly before recurrence, retain exact signs and first-index argmax. This avoids certifying a rounded sum. Save rational input/output values, rank gaps, gate zero counts, truth/prediction and exact margin. Run at most16 selected cases per <=20CPU chunk. Frozen controls cover positive/negative ReLU branches, full-volume output ties, equal hidden/readout rows, strict input ranks and exact input ties with explicit stable policy. No random inputs and no fitting.

Competing explanations: (1) output clipping has a semantic error, exposed by exact mislabelled witnesses; (2) floating direct evaluation is unstable near a boundary, separated by Fraction recurrence; (3) error volume bias comes from mass integration despite correct captured point semantics, investigated separately by T131. No claim that interior-point checks prove the full simplex, absence of overlap, or complete error-set coverage. Input ties are separate measure-zero ambiguity, not silently counted as successful strict interiors.

## Status
Bounded floating and exact point checks complete; independent source/raw arithmetic review complete ea53df3.

## Results
All35098 fixed ray-sum witnesses match parent second-argmax and classify incorrectly; no input ties. All64 frozen smallest-margin cases pass exact Fraction recurrence on unrounded sums of stored binary rays, with zero floating/exact label differences and zero exact gate ties. Exact truth-minus-best-other margins range from-.002063120414889303 to-.00043188188539797093. Six hand-computed ReLU/tie/rank controls pass. Full witnesses in floating.npz; rational inputs/logits/signs in exact_0.json through exact_3.json.

Commands from repository root: `.../python m16/approaches/error_cone_semantics/check.py controls`, then `floating`, then `exact 0` through `exact 3`; use `/workspace/clean-neurons/.venv/bin/python`, one-thread OMP/OPENBLAS/MKL settings, review flock and `ulimit -v 2500000`. Each chunk runs independently under20CPU. Source/archive hashes and exact selection are saved before rational runs.

## Compute accounting
One thread,2GB; primary cycles<=110CPU, review cycles<=20CPU. Every35098 deterministic forward pass and64 exact rational forward passes count as review compute, not estimator runtime. Imports, serialization and checkpointing reported. No Gaussian integration or random model inputs. No cost-competitive claim. Floating pass1.12191CPU; four exact chunks together4.59148CPU, total5.71338CPU plus tiny controls (0.031793649CPU; total5.745178159CPU in summary.json). Each timer includes imports and previous persistence; final JSON serialization/return overhead outside timer. PeakRSS recorded in summary.json;35098 floating plus64 exact trained forward passes.

## Claims
[inferred] A single exact incorrect semantic witness can falsify the captured label/error interpretation locally. [measured] No semantic error is exposed by all35098 frozen floating witnesses or64 exact rational checks. This supports the fixed point interpretation only; it is not a proof of whole-cone correctness or mass completeness.

## Alternatives not ruled out
Errors elsewhere inside simplices, degenerate simplices without genuine interior, overlapping/missing error regions, input rounding, other models/parents, exact checks beyond64. Full affine-logit certificates or independent complete clipping would be stronger and require a separately claimed follow-up.

## Failure modes
Floating sums can move across thin decision boundaries. Exact arithmetic applies to the stored binary geometry, not ideal mathematical rays. First-index output ties can occupy full volume; equal input ranks require explicit separate reporting. A passing point panel does not establish full accuracy.

Independent T55 review ea53df3 checks all35098 saved ranks/labels and exact64 smallest-margin selection, then independently reproduces every selected input, logit, gate sign and incorrect label using integer-scaled recurrence. This strengthens the finite point claim, not whole-cone coverage. Reviewer preserved an initial repeated-NPZ-decompression resource failure and repaired materialization; its cost is separate from producer accounting.
