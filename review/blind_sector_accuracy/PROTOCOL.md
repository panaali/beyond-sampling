# T268 fixed H4T4 blind-sector numerical discriminator

Claim57f443c6, after completed T278. This is one fixed synthetic-control plus saved-reference contraction, not another model/mass run. No payload has been decoded during source preparation.

## Probability and scope

For four independent centred Gaussian inputs, each strict sign orthant has mass1/16; coordinate-zero boundaries have measure zero. The at-most-one-positive event comprises masks0,1,2,4,8 and has mass5/16. These probabilities are not network error rates and do not transfer to T10 (whose corresponding mass is11/1024).

Consume the pinned exact H4T4 correct-class simplices and the corresponding T72 per-simplex outward quadrature intervals and truncation upper bounds. Source `check_enclosed.py` iterates the stored correct-simplex sequence in order and binds its compressed SHA. All5,058 rows are retained in that order. The original exact coverage/disjointness and numerical/platform assumptions remain inherited premises, not newly proved by this aggregation.

Each ray matrix must be4x4 integers, with no coordinate having both signs, no identically zero coordinate, and nonzero exact determinant. Coordinate zeros on rays are allowed. The cone interior has the unique sign of the nonzero entries in each coordinate. A separate24-permutation determinant reproduces each stored positive L1-normalized determinant. No stored gate/history or new forward is needed to assign the orthant.

For each component, convert the saved binary64 qlo,qhi,epsilon exactly to Fractions and take [qlo-epsilon,qhi+epsilon]. Intersect with [0,1/16], justified by nonnegative mass and whole-simplex orthant membership. Sum intervals exactly per orthant and intersect with its known1/16 mass. Preserve unclipped sums and raw global interval; assert the latter is inside the original broader floating global enclosure. No midpoint substitutes for an interval.

For sector mass p and correct mass c in [L,U], error mass lies in [p-U,p-L], conditional accuracy in [L/p,U/p]. The optimal deterministic **binary correctness indicator** using only the sector has loss f(c)=min(c,p-c). Its lower bound is min(f(L),f(U)); its upper is p/2 if [L,U] crosses p/2, otherwise max(f(L),f(U)). Sum these losses over the five blind sectors. Strict0<L<=U<p proves both correct and incorrect positive-measure subsets within that sector under the inherited premises.

This tests only sector-only deterministic correctness sufficiency. The true second-rank label varies within a sign sector: even a network with a constant predicted class can have nonconstant correctness there. Therefore a positive binary-correctness loss is NOT a predicted-class quotient obstruction. The original C8/T268 conditional predicted-class-law objective remains unresolved by this numerical baseline. A nonconstant correctness indicator does not rule out a compact conditional-probability law or computing total accuracy without reconstructing individual labels. We do not infer full predicted-class laws, a mechanism-derived sector estimator, a general quotient impossibility, or a T10 result. The oracle enclosures are not free: the complete historical reference cost/advice is disclosed, and there is no claimed reduction in construction cost.

## Frozen controls and coverage

Four valid orthant fixtures: positive/negative diagonal, sole-first-positive, and a triangular boundary-ray case. Four guards: mixed coordinate signs, missing coordinate span, singular simplex, wrong shape. Six exact probability-law fixtures cover zero/all correctness, below/above/crossing half and exactly half. Three malformed intervals reject negative, inverted and above-sector bounds.

All17 controls must pass before target archive decoding. Then read exactly5,058 rows in20chunks (19x256+194), all16 sectors in ascending mask order, and the five fixed blind sectors. Success has76 progress events,5,066 orthant attempts and25 binary-law attempts. Expected determinant evaluations are5,063 including the singular control; there are121,512 permutation terms. These are selected construction counts, not native FLOPs.

## Resources and durability

One process, one thread, review flock `/workspace/work/codex-review.lock`;17internal19soft20hardCPU,2GiBAS,25wall seconds after obtaining lock. No automatic repeat or budget increase. Source/resource gate and immutable owner execution gate must be published in separate successful steps before launch.

Every control starts durably; returned values/reasons are saved before assertions. Every target row has an atomic active record before geometry arithmetic and after its determinant/value. Completed scalar records append and flush once, preserving partial chunks if stopped. Chunk hashes, counts, full source pins and failures are retained. Existing output/log refusal prevents overwrites. No owner module, NumPy, checkpoint loader, candidate/model, RNG, face law, geometry generation, Gaussian quadrature or mass evaluator is imported/called.

## Independent review and interpretation

T55 accepts source/theorem/resource review. h4's explicitly delegated saved audit will independently check synthetic controls, exact sector assignment, stored mass/interval arithmetic and coverage with separate code; source/resource gate is required before its bounded run. Repeating determinant/aggregation arithmetic is review compute, not a model/mass replay.

Report all16 sectors and all five blind sectors, including degenerate/full/unknown intervals. No sector, model, threshold or outcome is selected after viewing results. No HP1/sealed truth is used. Any apparent negative is scoped to a constant correctness indicator on these five sign sectors; richer state and conditional probability summaries remain open.
