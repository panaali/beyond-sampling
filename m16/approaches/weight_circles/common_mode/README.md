# Common-mode correctness discriminator

## Question
Does a common shift x+c*ones preserve correctness on the existing two T187
trained models? Such invariance is necessary to drop the common input offset
and keep only the natural zero-sum projection. It is not necessary for every
other low-dimensional projection or conditional representation.

## Method
Scope9df81752 precedes implementation; `PROTOCOL.md` freezes all768 saved
witnesses and four analytic controls. `core.py` reconstructs exact symbolic
preactivation/logit rows, intersects the open gate-valid common-shift interval,
and partitions every pairwise logit crossing. First-index ties are preserved.
Only full-vector identical winning rows permit a robust permanent tie.
Exact positive cube radii certify actual stable correctness at each witness.
Two robust opposite probes have the same projected coordinates and certify a
common open neighborhood of projections, avoiding a null-line-only inference.

## Status
Source/theory independently cleared3ee0667b. Four controls pass4b3886c1 in
0.004423146CPU, saved audit pending. Trained driver505220d3 is source-only,
unexecuted, and has no execution gate. No trained forward or reference draws.

## Results
Analytic H2T3 crossing case has gate interval(-1,infinity), true label1 and
margin1-c; correct status changes atc=1 while earlier label0 owns that tie.
The invariant control and allzero permanent tie show no change. Distinct
full rows tied only on a common line correctly receive no robust certificate.
The original circle-law phase is separate, in the parent folder.

## Compute accounting
Four exact synthetic controls only, with per-expression integer/rational
arithmetic counts and maximum integer bit sizes. Gcd, comparisons, hash/index,
control, allocation, bit complexity and full runtime are not priced by those
counts. Named source counts are not a complete estimator cost. Trained saved
contraction, if independently gated, gets one review-lock process capped20CPU,
2GB and one thread, with16CPU graceful stop and durable partial artifacts.

Commands (one-shot, refuses overwrite):
```
ulimit -v 2500000
ulimit -t 20
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock timeout 25 /workspace/clean-neurons/.venv/bin/python m16/approaches/weight_circles/common_mode/controls.py
```
No automatic continuation or expanded sweep on failure.

## Claims
[inferred, independently source-reviewed] Positive cubes with opposite
correctness at common-shift partners reject this natural projection beyond
a single ambient-null line. [measured, saved review pending] Four controls
pass. No trained result or mass/cost saving is claimed yet.

## Alternatives not ruled out
Another projection; conditional decomposition; approximate invariance on most
probability mass; or a useful weight-derived orientation without exact reduction.
A finite absence of witnesses proves none of these. Nonzero margin slopes
alone may be irrelevant because correctness can stay fixed.

## Failure modes
Treating a gate endpoint as interior; missing rival-rival crossings; unstable
ties mistaken for permanent ties; output-label variation mistaken for correctness
variation; ignoring input ranks; finite probes mistaken for a probability law;
or extending a local failure to every possible circle representation.
