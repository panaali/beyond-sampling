# T463: local mass/pruning soundness gate

## Question

Can T455's floating mass charge and pruning decisions support a certificate, and can a conservative local checker repair the exposed obligations?

## Method

Frozen producer: `4e29c33861a0983535cd9bc80d660cfd9f2dd412`, `origin/claude/t455-late-triple-bound`. Read `rmass.py`, `cbab.py`, `late_triple_verify/net.py`, the T459 readiness review, T460 halfspace review, and repository literature B9 (Jia–Rinard): floating solver/verifier outputs need independently checked safe bounds. No new ARC sources, trained models, truth values, BaB search, or author edits.

`PROTOCOL.md` was published before evaluation. `controls.json` freezes44 mass cases (SHA256 `ccc0acdd83c9b5aaecee0a75f5d8d169665f709b9c07262dfc037dcdfd4458df`); `generate_controls.py` records their construction. `local_verifier.py` supplies independent Arb mass bounds and exact-rational local CROWN bounds. `run_controls.py` compares producer outputs and runs exact 1-D activation-region controls. `REFINEMENT.md` discloses the post-first-pass improvement to the true-mass lower-bound witness; first results and implementation are preserved in `first_pass/`.

## Status

Bounded phase complete; [independent cross-review](../is_v4_contract/T463_REVIEW.md) passes the scoped local proofs and reproduces the defect on an off-suite box. Six actual mass counterexamples established. Local primitives remain separate from producer adoption; ordinary CROWN remains uncertified, and no certificate is promoted.

## Mathematical obligations and local repairs

**Mass representation.** In coordinates x=g*y(u), Jacobian is g^9. With y_p3=r, y_p2=r+1, y_p1=r+1+a, and seven y_l=r-l, integrating a and l absorbs eight powers of g, leaving g/(2pi) times the two fixed Gaussian densities and eight Phi differences. On each r/g rectangle, g<=g_hi and g² q(r)>=g_lo² min q, where q(r)=2(r+1/2)²+1/2. Each Phi difference is at most both (i) Phi(max z_hi)-Phi(min z_lo) and (ii) g_hi*(coordinate width)*max phi over the enclosing argument interval. Thus their minimum and the product of nonnegative factor bounds give a valid upper Riemann sum. A lower-sum control instead bounds every factor below on an included subregion, so exceeding the purported upper establishes actual mass underestimation.

**Tail.** For g>20, ||y||²>=r²+(r+1)²>=1/2 analytically. Integrating g^9 exp(-m*g²/2) gives 16*m^-5*Gamma(5,m*20²/2). Multiply by box volume/(2pi)^5, with Gamma(5,w)=24*exp(-w)*sum_{k=0}^4 w^k/k!. The local repair fixes m=1/2, sacrificing tightness while removing the unverified float `net.minsq` assumption. T455 instead passes that float lower bound, whose rounding is not certified.

**Arithmetic.** All supplied binary endpoints are converted exactly to rationals and then enclosed by Arb. Grid positions are exact rational interpolants, so they cover the original endpoints without float cancellation; min/max and elementary functions propagate enclosures throughout. Arb's upper endpoint is converted to float and then moved with `nextafter(+inf)`, since nearest-rounded `float(tot.upper())` alone is not a guaranteed upper float. Invalid shape/domain/nonfinite inputs or grid settings reject; zero-volume boxes return0; unresolved nonfinite output raises. This is a local upper-bound proof conditional on the documented Arb enclosure contract and correct implementation, not a sampled proof or end-to-end T455 certificate.

**Pruning.** The event is the conjunction of nine margins <=0. Discarding a box requires a verified strict positive lower bound for at least one margin. A heuristic slack on a halfspace cannot fix an unadjusted pruning predicate. `exact_crown` recomputes interval preactivation bounds and a backward affine bound using exact rational arithmetic: stable ReLUs are exact; unstable positive backward coefficients use a valid lower line (0 or identity), and negative coefficients use the exact upper secant h/(h-l)*(z-l). Recurrent/input affine propagation and box minimization are exact. Induction over backward substitutions proves each returned bound is <= the true real-arithmetic margin everywhere in the box. Submitted upstream bounds are not trusted. Original output rows are subtracted exactly rather than using T418's pre-rounded `WO[others]-WO[p3]`; inputs use exact original WI and coordinate map. Thus the target is the real network with given dyadic weights, not automatically the pre-rounded UNet or machine float inference semantics.

**Remaining global obligations.** The trained model, coordinate/export identity, and actual pruned boxes need replay under a verifier; an ordinary CROWN or alpha-CROWN positive bound remains uncertified. Parent/child box coverage, no dropped frontier cells, any checkpoint/model identity, root-domain tail, and the sum of upper charges also need outward arithmetic. For root U=4,t0=2.5 the source's tail formula is 2*sf(10)+8*sf(10/sqrt(2)), from a union bound on one coordinate and eight coordinate differences; its float evaluation is not a verified upper endpoint. Summing verified doubles with `RC.sum()` is likewise not an upper-bound operation. No amount of sampled pruning validation establishes these universally.

## Results

[measured] Six of44 cases are true mass underestimates: thin-a k=54,55,56,60,80,100. Both128/256-bit checks agree. At k54, a=[2^-54,2^-53], r=[-.75,-.25], lows=[0,1]^7: producer float charge is **5.621043391498192e-54**, while the verified mass on a positive subregion is **>4.36452e-31** (128-bit lower endpoint, conservatively shortened for display). The independently repaired coarse upper is1.3006153634512807e-23. The same actual underestimate persists at24x48: repaired upper1.0606858564302765e-24, producer float unchanged. Root independently supplies a rational-only lower bound2.241137083307659e-32, using e<3 and pi<22/7, in [T463_ROOT_WITNESS.py](../../docs/T463_ROOT_WITNESS.py).

[inferred, exact binary mechanism] Both1+2^-54 and1+2^-53 round to1 in float64; the a-width inside `_diff_sup` becomes0. Its minimum with the density-times-width candidate therefore makes the body0. Multiplicative1e-9 inflation cannot restore an erased factor; only the tiny radial tail remains. This refutes the general claim that the source charge is always an upper bound, but does not establish that the actual unpublished frontier encountered these adversarial boxes.

[measured] All44 repaired upper endpoints pass upward-conversion and subregion-lower checks at both precisions. All19 exact-pruning toy cases pass independent activation-region minima checks (30 exact regions total); two cases admit a valid positive prune. The deliberately unsafe positive proposal is refused by independent recomputation. Seven malformed/domain/grid controls reject. These checks support implementation correctness; the mathematical arguments above, with their stated contracts, supply the reason for conservativeness.

[measured] The first pass detected the same six upper-expression discrepancies but its whole-box lower witness was too loose to prove true underestimation. That failed witness, first outputs, and exact implementation are preserved. The successful subregion refinement was disclosed and published before rerun; the44 input boxes were unchanged.

## Reproduction

From the repository root, with NumPy/Torch and python-flint available:

```bash
mkdir -p /workspace/work/bound-soundness
git show 4e29c33861a0983535cd9bc80d660cfd9f2dd412:m16/approaches/late_triple_bound/rmass.py > /workspace/work/bound-soundness/producer_rmass.py
ulimit -v 2500000
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/workspace/work/bound-soundness/deps /workspace/clean-neurons/.venv/bin/python review/late_bound_soundness/run_controls.py /workspace/work/bound-soundness/producer_rmass.py
```

Only the small python-flint wheel was installed into scratch using `UV_CACHE_DIR=/workspace/work/bound-soundness/uv-cache uv pip install --python /workspace/clean-neurons/.venv/bin/python --target /workspace/work/bound-soundness/deps --no-deps python-flint`. No full environment or worktree was copied. Missing `/usr/bin/time` setup error is retained in `setup_failure.log`; Python resource accounting records CPU and peak RSS.

## Compute accounting

One CPU thread, <2GB per process; total phase cap600CPU-s. First suite2.743CPU-s after imports and refined suite2.696CPU-s, total5.439CPU-s for numerical bodies; peak RSS214364KiB (~209.3MiB). Bash process accounting including imports totals8.850CPU-s (first4.679s, second4.171s). Dependency installation/setup and manifest generation are additional minor runtime. Python3.12.14, NumPy2.5.4, Torch2.14.1+cpu, python-flint0.9.0. Zero neural-model forwards, LP solves, trained-model loads or BaB cells. Synthetic exact piecewise recurrence and mass elementary-function evaluations are charged as CPU time. Strict F<=1 and lenient F<=10 estimator bars are not applicable: this is a soundness/control task, not an estimator comparison.

## Claims

- [inferred] Floating inflation alone is not a proof of mass/pruning soundness; endpoint transformations can lose widths before the inflation is applied.
- [inferred] The local Arb and exact-rational constructions have conservative mathematical proofs under their explicitly stated arithmetic/model contracts; independent implementation review passes the scoped primitives; integration review remains required.
- [measured] Six frozen boxes violate the purported floating mass upper bound, supported by independent interval and root rational witnesses. No measurements on the producer's actual frontier or trained model are claimed.

## Alternatives not ruled out / exhaustion

Tried: ordinary/dyadic/random boxes, thin-a/low/one-ULP r/zero-volume controls, 128/256-bit checks, two grid sizes, exact toy-network activation enumeration, invalid inputs. Initial true-mass witness was too crude; refined subregion retained separately. Not tried: author's 48x96 scored validation, recorded frontier recovery, trained-model verifier throughput, exact rational alpha optimization, interval-CROWN acceleration, outward-rounded trained-network replay, full BaB bookkeeping, or all-triple certificates. Source availability is not evidence about unpublished runs.

Ordinary boxes may be well covered by the heuristic inflation, while deep cancellation cases fail; finite failures do not establish that the author's actual frontier reached those boxes. A bound-expression discrepancy can also reflect grid choices or looseness rather than true underestimation; only a true-mass lower witness settles that distinction. Exact pruning may remain too loose or costly to help late triples even when sound.

## Failure modes

Arb can return uninformative balls at insufficient precision; the upper remains conservative when finite but may be useless. Rational numerators/denominators can grow with model size and depth; no scalability claim is made. Real-network proofs do not enclose floating inference automatically. The primitive does not validate external model provenance, search coverage, or summed probability. This phase ends by scope, not a compute-budget impossibility; more computation is useful only after those integration and provenance obligations are specified.
