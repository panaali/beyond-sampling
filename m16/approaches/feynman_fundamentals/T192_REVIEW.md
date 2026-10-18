# T192 independent saved/source review at 9d2feb68

The saved port totals and stated ratio arithmetic reconcile. The description “complete FLOPs” is not established by `audit.py`; its 1.37 multiplier must remain a model-dependent estimate, and its transfer to H8T5 is unvalidated. This review imports no author code and runs no network, geometry, mass integration, GPU workload, or Monte Carlo.

## Evidence checked

`check_t192_saved.py` pins every file in the frozen author folder and independently checks 30 completed run summaries against 7,488 unique per-half records. Every run covers its stated half indices once. All saved count sums and mass sums reconcile. Where both geometry and thin-simplex counters are present, the original per-simplex proxy formulas reproduce the recorded mass charge exactly. Ten audit tables conserve their declared site sums. Their site-attributed charges each exceed the main estimator's counter by 57,120 cold-setup units; that difference is retained rather than silently equating the two totals.

All ten H8T4 seed/mode pairs preserve historical ops, cells and resolved counts. Both H8T5 seed0 modes preserve ops, cells, cut and resolved counts. Accuracy offsets remain finite: about 8e-12 at T4 and 3e-10 at T5. The source copy `estT_base.py` is byte-identical to the shared `estT_fix/estT.py` at review time.

The first reviewer pass additionally asserted identical historical T5 simplex counts and failed on recipe108: GPU 2,696,927 versus historical 4,893,634. The failure is an audit-premise correction, not a suppressed source discrepancy. The historical `run5.py` optional leafsum wrapper calls `mass(c, Ops())` for every leaf and increments the same global simplex counter while discarding its Ops instance. The frozen historical recipe record includes leafsum metadata. The repaired checker retains both simplex counts explicitly and limits the parity claim to the matching fields. No estimator was rerun.

## Cost scope and concrete gaps

* The determinant, inverse and SVD formulas are nominal algorithm models, not installed implementation counts or proved upper bounds. For example `2*n**3/3+n` ignores pivot selection, absolute values, comparisons, and the actual determinant path. Tiny matrices do not justify dropping lower-order work under a convention that counts those operations.
* `audit.py` disables `T192_FAST` and audits the Python/NumPy implementation. The accelerated port's `geomfast._rank` calls `np.linalg.svd(M)[1]`, requesting singular vectors as well as values; the audit uses a singular-values-only cost formula. GPU mass kernels also differ from the audited scalar program, including omission of the scalar four-dimensional inverse/correlation preamble. Thus this is not an instruction census of the timed port.
* Explicit source work is missing from the advertised complete tally: crossing-ray incidence `c.T[a] & c.T[b]`, surviving clipping normals `sign*n`, initial chamber arithmetic and initial `g1`/`A1`, and scalar mass/error accumulation. `_adj_full` executes multiple `Z.sum()` reductions but its initial `2*len(c.N)` allowance prices only one AND/reduction combination. `_cell_error` charges one unit on the zero-gate branch despite evaluating the vector `g.sum()`. Some other formulas over-allow work, so these observations do not supply a corrected total or prove the net underestimate.
* `summary.py` uses exact per-case audit records only for the five clean H8T4 recipe cases. Every other completed pair, including every H8T5 pair, uses literal `AUD_FACTOR=1.37`. The H8T4 ratios from the ten declared audit tables range 1.3582–1.3857; this finite range is not an H8T5 envelope. `audit._orthant` has no d=5 branch and raises `ValueError(5)` after the underlying evaluation, so the posted audit does not cover T5.

The saved formula gives H8T5 seed2 ratio 3.403883 under the old proxy convention and 4.555803 with the transferred factor/T196 denominator. Those arithmetic results reproduce, but the latter is conditional on the transferred cost model. H8T4 clean worst ratio is 0.361419 under its declared audit model. No complete native-cost or universal accuracy conclusion follows from this review.

## Port equivalence and geometry limits

Buffering masses is structurally legitimate because upstream geometry and recipe decisions do not read returned Gaussian masses; zero placeholder mass returns are accumulated only into the eventual error. Batched cofactor formulas and scalar formulas represent the same real-arithmetic expressions in dimensions four and five. Floating determinant branch selection, operation association, GPU reductions, and final summation differ. Numba uses a different SVD call and returns faces in discovery order whereas the Python implementation iterates a set of faces. Finite saved parity supports these specific executions; it does not prove exact geometry, numerical stability for every cone, or identical ordering on every input.

Per-half JSON records do not include source/runtime/model byte hashes or saved ray matrices. The source loader and matching historical counts support provenance, but cannot retrospectively reconstruct those missing bindings. Current `estP.run` can resume from an existing log without checking its model/source/configuration identity. All 30 reviewed completed logs have unique full half coverage; that does not certify the general resume mechanism. Raw rays required for independent trained-cell T5 mass checks are absent from this frozen tree.

## Monte Carlo offset

All 22 saved MC records reproduce integer hits/n, point accuracy and standard error. Eighteen belong to the original stream family; fifteen have a completed exact run in this snapshot. Twelve of those fifteen MC-minus-exact offsets are positive, with largest absolute z about 1.980804. The original code reinitializes the same seed (12345) for each model, so these are correlated observations; a binomial sign test or independent pooled z calculation would be invalid.

Four independent-seed recheck records are present for H8T4 seeds0–3, using 9400–9403. Their z values are +0.696955, −1.561047, +0.884466 and −0.093227. This partial recheck does not establish a systematic low bias or rule one out. The snapshot contains totals, not per-block RNG states/hashes, so saved arithmetic review cannot independently certify execution coverage or stream disjointness. MC intervals also do not certify the quadrature's error on individual cells.

## Accounting and status

Reviewer CPU is recorded in `t192_saved_review.json`. The failed first reviewer assertion and its specific repair are retained above; no author workload was repeated. T6/orth6 source validation and unpublished later runs remain outside this bounded review. Claude DRI retains acceptance and any revised cost/port implementation decision.
