# Exact synthetic metrics controls — source preparation only

Scope8610f317. Fixed8 cases in fixtures.json and10 invalid-input guards; no
model, observed target table, face, quadrature, random assignment or array work.
These finite tables are equally weighted generic arithmetic controls, not actual
balanced-kind co-area samples. No population probability or estimator inference.

Cases: constant; allzero; independent label with95% positive square coverage;
perfect outcome label; unseen validation label with pooled training fallback;
exact cutoff ties; kind-preserving fixed label permutation; actual smaller-array
tail reselection. Preserve all fitted means/residuals, undefined zero-baseline
ratios and both old-tail intersection/new-tail outputs. All expected rational
scores are written before author calls. The permutation is explicit, not random;
it tests multiset conservation, not the future random-assignment null law.

Guards: empty, negative, NaN float, duplicate identity, capped status, missing
label, cross-kind permutation, duplicate permutation, zero tail fraction and odd
kind population for half selection. All must raise ValueError, with raw error or
unexpected return saved before assertions. Guard rejection means no result may
be inferred; no dropped records or replacement zeros.

Primitive score pools over all A rows; fits exact per(kind,label) means; predicts
all B including unseen fallback; returns mean squared residual, baseline squared
residual, centered residual variance and undefined ratio when baseline0. Tail
sorts by decreasing exact square then kind/index; ceil(q*N) rows are kept.
Half uses contiguous original indices0..n-1 within each kind and keeps firstn/2.
No assumption of real Sobol prefix equivalence is tested. Label permutation only
moves labels, preserves kind and is an explicit bijection. Source validation,
Fraction arithmetic and Python allocation are real diagnostic overhead.

Sole later run requires a published independent source/resource gate and a fresh
immutable execution manifest binding this protocol, source, fixtures, prior
completed design review and current source review. Design review explicitly did
not authorize execution. All pins checked separately, reviewer complete/pass_all
and exact source pins checked. External20CPU/2GiB, internal17CPU, one thread,
review lock; pinned venv executable/prefix, no NumPy import. Refuse existing
output/log, durable pre-call attempts and raw results before equality checks.
Expected39 progress records, one immutable attempt; failures preserved, no
automatic retries. Main timer excludes top-level imports and final save work;
external cap covers process. Independent saved arithmetic audit follows. No
full-target data/label application or new random null is authorized here.
