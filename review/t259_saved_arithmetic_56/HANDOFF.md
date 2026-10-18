# T259 independent saved exact arithmetic — complete

The sole bounded audit passes all17fixed cases,36systems,35nonsingular solutions
and the one expected singular system. All returned row fields, literal fixture
bindings, exact objectives/labels/routing,36producer events,36reviewer events and
source/runtime/resource pins reconcile. No candidate, model, trained-array,
Gaussian or RNG calls were made. `output/results.json` retains every independently
reconstructed row and all36exact system determinants.

The independent method uses permutation determinants and Cramer's rule rather
than candidate Gauss–Jordan elimination. Pair systems are built from the
pair-Laplacian/Kronecker identity rather than expanded sample-by-pair features.
Each unique coefficient matrix also satisfies its exact normal equation.
Canonical rational serialization and strict recursive types are checked without
numerical tolerances.

## Observations

- Rectangular/rank-deficient readout, common mode, shared intercept and fixed
 cell/pooled routing commute under the stated common-map assumptions.
- All eight pair convention × sample sum/mean combinations give the same
 centered coefficients with matched penalty scaling. Every pair objective is
 reconstructed independently.
- Unmatched penalty changes the frozen query's first-index label0→1.
- Output-specific scalar penalties give1+1/2, versus2for the directly fitted sum.
- The singular case has exact zero determinant; explicit nullspace choices agree
 on training while differing on the query. It is an expected exception, not an
 unreported campaign failure.
- ReLU clipping changes the fixed query label1→0, retaining the theorem boundary.

These are exact synthetic observations under the independently reviewed prior
proof. They do not establish numerical-program bit parity, trained agreement,
a99%ceiling, insufficient features or a negative result for weighted/top-rival,
nonquadratic or recurrent objectives.

## Provenance and compute

Owner source06c7cf14, doc-only22b2f8e4, source review78ed0a7b/shared54a92781;
producer immutablec3dd1878/shared42239a27 preceded sole rawa772c30a/shared3d873cc2.
Reviewer assignment7459ae50 precedes preparation. Checker32ecbf10 was frozen
before output decoding; coverage/CPU headline arrived after local freeze.
Pre-run770d4d23 adds explicit producer CPU/RSS metadata guards only. No scientific
formulas or fixtures changed. Hash-only completed inputs a5617782 retain the
original source freeze and exposure chronology.

Reviewer source gate64ef1de3/shared1eab3b3e and immutable90e34439/sharedf8eb4b96
were successfully published before the separate sole launch:
`bash review/t259_saved_arithmetic_56/launch.sh`.

Reviewer CPU0.168621204s, peak21,884KiB; external log0.20wall/0.18user/0.02system.
Actual19soft/20hardCPU and2GiB caps,17internal/25wall-after-review-lock. Producer
CPU0.787153895s remains separate. No unexpected failures, retries or candidate
reruns. Preparation/source review and native fit-cost interpretation are not
silently included in the tiny arithmetic timing.

Original README/protocol/source files remain frozen as pin evidence; this new
handoff supersedes their preparation status. Owner78 may attach the independent
result and close the bounded numerical phase. DRI registration/confirmation and
any separately prioritized trained comparison remain open.
