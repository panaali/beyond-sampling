# T463 frozen bounded protocol

Source pin: `4e29c33861a0983535cd9bc80d660cfd9f2dd412` (`origin/claude/t455-late-triple-bound`). Review only; no author edits, model checkpoints, truth access, or BaB search. CPU slot 2, one thread, <2 GB, <=600 CPU seconds across this phase. Python dependencies from `/workspace/clean-neurons/.venv`; if needed install only python-flint into scratch. Protocol published by root before numerical work.

## Frozen checks and pass criteria

1. **Source/proof audit:** derive mass formula, nonnegative factor sup bounds, radial tail, required endpoint coverage, norm-lower-bound and sum rounding conditions. Trace pruning and coordinate/network semantics. Jia–Rinard takeaway (repository `lit/novelty_review.md`, B9): float optimizer/verifier output requires independent safe bounds; finite sampled checks do not certify.
2. **Mass controls:** exact binary endpoints; 8 baseline boxes (root and seven deterministic dyadic subdivisions); 16 pseudorandom boxes with NumPy generator seed 463 (r in [-4,4], eight other coordinates in [0,4], sort independent uniform endpoints); 12 thin-a controls with a in [2^-k,2^(-k+1)] for k in {20,40,50,51,52,53,54,55,56,60,80,100}, r in [-.75,-.25], seven lows in [0,1]; 8 thin-low controls with one low in [1, nextafter(1,+inf)] or [0,2^-k], k in {52,54,80}, plus exact-zero-width and narrow-r endpoint controls. Record the actual generated suite and hashes. Compare frozen float and producer arb implementation on 4x8 grids for all controls, then 24x48 grids on the root and first failing control (or first thin-a if no failure). Float-vs-arb failure means float < rigorous lower endpoint of the *same conservative expression*, or where available a positive enclosed subregion integral lower bound exceeding float. Comparing upper estimates alone is explicitly only a diagnostic. Arb precisions 128 and 256 bits; retain undecidable cases, no tolerance erasure.
3. **Local mass repair:** independently implement interval arithmetic per-factor upper Riemann bound, exact binary endpoints, covered grid, universal m=1/2 tail, and upward float conversion. Proof obligations plus all valid-suite controls must pass; malformed boxes/m/nr/ng must reject or return unresolved. A finite test pass is not the proof. No global certificate claim.
4. **Pruning repair/control:** implement an independent exact-rational local RNN CROWN verifier (forward exact interval bounds, backward exact linear ReLU relaxations, exact coordinate and output differences). Frozen synthetic controls: cancellation at 2^53; near-zero ReLU; 16 seeded small 1-D, 2-step, width-2 recurrent nets (seed 1463; integer coefficients from -2..2 scaled by 1/2; box [-1,1]). Exhaustively enumerate 1-D exact activation breakpoints and margin minima; every verified lower bound must be <= exact minimum, and every verified positive prune must have exact minimum >0. Include a deliberately unsafe submitted positive bound to verify rejection/fallback. This supports the generic local primitive only: no trained-network replay or ordinary CROWN certification.

## Alternatives before results

- Inflation may cover ordinary boxes but fail after cancellation of transformed endpoints; thin-a vs ordinary controls separate this cheaply.
- Arb disagreement can reflect grid differences or a loose upper bound rather than an actual mass underestimate; use covered identical rational grid and an enclosed positive subregion integral lower bound to separate them.
- Exact pruning may be sound yet too loose/slow to rescue late triples; report this limitation, retain ordinary CROWN uncertainty, do not infer feasibility from synthetic controls.

## Stop/report

Stop on 600 CPU seconds, memory cap, or completed controls; preserve failures and remaining checks. Report commands, versions, hashes, CPU/RSS, variant exhaustion and scope. Root arranges independent cross-review before any adoption. F<=1 and F<=10 estimator bars are not applicable to these soundness controls.
