# Independent final-state replacement specification

This is an executable-contract specification, not execution permission or a new numerical result. It reuses the reviewed T258 endpoint identity and adds exact feature ordering, precision boundaries, fit dependency counts and required deployable artifact schema. Prior source/summary audits remain authoritative for historical records. No historical fitted payload has been recovered here, and no global absence claim is made.

## Fixed target and dependency closure

The target is the source C2 `best` all16 endpoint on the same H16,T10 checkpoint. With every final neuron overwritten, the endpoint is `argmax(ReLU(float64(float32(F_10))) @ Wo.T)`, first-index output ties. Earlier replacement steps do not enter this endpoint. The original network supplies clean final preactivation training/validation targets and reference labels. This is learned final-state prediction, not recurrence reconstruction. Float64 weights/readout here specify the proposed reproduction computation, not a verified historical NPZ payload dtype. Actual stored Wi/Wh/Wo dtypes and exact canonical conversion remain the separate model-binding prerequisite. The endpoint identity requires the same realized Wo and readout semantics; no historical bit equality is inferred.

A mathematically matched endpoint needs nine comparator-bank models (steps1..9, four joint targets indexed [1,6,7,8]) plus twelve final-step candidate models (each jointly predicts16 targets). Thus there are **21 fitted cell-regression model objects**, not21 linear-system solves: each object includes one pooled solve and one solve per observed cell. Actual cell counts/solver calls must be retained. Step10 bank output and candidate families at steps1..9 are unused and omitted. This does not reproduce partial-replacement interventions or their evaluation-ranked neuron order.

Bank models use the A123K/m5rest design, fit to clean original preactivations. Each produces ReLU of a float64 prediction, then is cast to float32 before use as a hinge feature, exactly as C2's stored Htr/Hva/Hev. Concatenating with float64 base features promotes the stored float32 hinge values to float64; casting after the final fit would define another recipe. Training bank predictions are in-sample, matching C2; this is not cross-fitting. Held-out bank features are predictions from training coefficients, never held-out true comparator values.

## Input-derived encodings

For prefix length t, rank inputs descending using the frozen NumPy `argsort(-x[:,:t], axis=1)` policy. Let m_k be rank k (zero-indexed), a_k=t-1-position, K=count(x>0), xc=x[t-1]. Missing m_k=0 and missing age=t. Default NumPy tie behavior is an explicit runtime contract, not a claim of stable sorting; finite input ties are retained and counted, never resampled. True target uses the separately pinned source `argsort(x,axis=1)[:,-2]`. Output argmax always uses first index.

Base columns, in order:

- m3: [xc,m0,m1,m2], d4.
- m3clock: m3 then [m0*cos(theta*a0),m0*sin(theta*a0),m1*cos(theta*a1),m1*sin(theta*a1),m2*cos(theta*a2),m2*sin(theta*a2)], d10, theta=2*pi/8.42 with source floating construction.
- m5: [xc,m0,m1,m2,m3,m4], d6.
- m5rest: m5 then sum(max(m[5:],0)), sum(min(m[5:],0)), d8; empty rest contributes0.
- all: [xc,m0,...,m9], d11, with missing values0.

Let R=t+1. Cells: none=0; a1=a0; a12=a0*R+a1; a123=(a0*R+a1)*R+a2; a123K=4*a123+min(K,3); a1234K=4*(a123*R+a3)+min(K,3); a12K=4*a12+min(K,3). No new discretization or feature standardization.

Final-step candidate order and dimensions are fixed:

| index | source family | cell | base and additional columns | d at t10 |
|---|---|---|---|---|
|0|L|none|m3|4|
|1|C|none|m3clock|10|
|2|A1|a1|m3|4|
|3|A12|a12|m3|4|
|4|A123|a123|m5|6|
|5|A123K|a123K|m5rest|8|
|6|A1234K|a1234K|m5rest|8|
|7|A1234K-all|a1234K|all|11|
|8|H1|a123K|m5rest then bank step9, comparator order[1,6,7,8]|12|
|9|H3|a123K|m5rest then bank steps9,8,7 in that order|20|
|10|Hall|a123K|m5rest then bank steps9 through1 descending|44|
|11|A12K-Hall|a12K|same44 columns as Hall|44|

## Cell regression and numerical domain

For each observed integer cell c, G_c=V_c^T V_c and B_c=V_c^T Y_c in float64. Solve `(G_c+lambda*(trace(G_c)/d+1e-12)I) A_c=B_c`, lambda=1e-6. Pooled G and B are sums of the unregularized cell statistics; solve `(G+1e-9*trace(G)/d I) A_pool=B`. Store sorted observed IDs and counts. Unseen cells use A_pool; seen cells use their own matrix. An all-zero pooled design is singular: preserve a solver-domain failure, with no silent pseudo-inverse or extra ridge repair.

An independently written implementation may use a different accumulation order, so exact historical floating equality is not promised. Exact rational synthetic Gram/target controls first establish the intended equations, then finite comparisons quantify arithmetic differences. Original source's 20,000-row prediction blocks and normal-equation fitting are numerical provenance, not a universal stable-solver claim. The new implementation must state its accumulation/solve/prediction order before any fit.

Validation targets are clean original final preactivations. R2_j=1-SSE_j/max(SST_j,1e-300), with the source's centered SST. Choose the first maximum family index separately per neuron over all12 candidates. A no-hinge comparison chooses only indices0..7 from the same scores; it adds no fits and must be fixed in advance. Evaluation predictions of each candidate are cast to float32 before selected columns are assembled. Convert selected matrix to float64, ReLU, then multiply by original float64 Wo.T. Do not fit logits, apply a shared family to all neurons, select on evaluation, or add nonnegative projection before the source ReLU stage as part of this baseline.

## Retained deployable object

A schema-versioned manifest must bind source/runtime/BLAS, checkpoint and converted Wi/Wh/Wo byte hashes, input streams, feature grammar/column order, theta construction, dtype/casts, ridge values and solve rules. Every one of21 fitted objects retains IDs/counts, full coefficient matrices, pooled fallback, dimensions and all fit status/timing/category counters. Retain all12 final candidate matrices and16 validation-selected indices, even if a later codec stores only needed columns. Bank objects retain all four comparator outputs and their fixed indices.

The deployable endpoint consists of raw-input encoding plus needed bank models, selected final columns, unseen-cell fallbacks, ReLU/casts and Wo. It does not need Wi/Wh for inference after fitting, but original weights remain provenance/reference dependencies. T260 must count routing, feature constants, table IDs, pooled fallbacks, reachable bank closure, precision and decoder framing; no bit total or minimality claim follows from object counts.

Raw evaluation retention must support independent integer reconstruction: x, true target, original/fitted logits and predictions, selected final preactivations, original final hidden/preactivation, all final candidate predictions, case masks, gaps and original packed gate patterns for D1. Coefficients and validation scores support independent selection reconstruction. Training and validation inputs and all bank/final targets/features need either immutable arrays or a precisely bound reproducible stream plus sufficient independently reviewable saved tensors; byte hashes alone do not establish numerical correctness. Default proposal retains inputs, needed clean targets, bank predictions, and coefficient sufficient statistics with chunk records. Large artifacts require a separately approved storage/resource plan, not silent omission.

## Scientific comparison

A new fit on new train/validation seeds changes fitted coefficients and selected families. The historical agreement .955395 is a retained scalar compatible with191079/200000 under the prior reviewed division convention, not a pinned deployable old-model score. T258's0.5pp point criterion is agreement in [.950395,.960395]; it is neither a universal hypothesis-test guarantee nor training-run variability evidence. Report conditional held-out agreement uncertainty separately and identify whether its interval is contained in that band. No sample-size adjustment after viewing accuracy.

Task accuracy and original-network agreement are distinct. Save the five disjoint outcomes: both correct; original correct/replacement wrong; original wrong/replacement correct; both wrong/same; both wrong/different. Their sum isN; disagreement is2+3+5, task gain is(3-2)/N. Keep exact counts/denominators and paired uncertainty distinct. D1 near union is g12<.05 OR g23<.05 OR at most one positive input, with overlap counted once. Pattern frequencies are self-inclusive evaluation-sample frequencies, not population cone mass. No gate/mass/capacity or universal feature-ceiling conclusion follows this baseline.

The T259 common-design ridge commutation theorem remains relevant: merely replacing a common-target quadratic hidden loss with matched all-pair quadratic margins is not a distinct rescue. It does not equate this per-neuron validation-selected/ReLU composite with a direct-logit fit. Weighted/top-rival objectives or richer features remain separately frozen future variants.
