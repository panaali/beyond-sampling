# T110: ordered top-two class families

## Question
Can exact second-argmax integration use quadratic rather than factorial initial families without changing accuracy? Claim1afb091 was pushed before implementation.

## Method
For ordered pair (a,b), a!=b, define C_ab by x_a>=x_b and x_b>=x_k for all other k. Split by sign(x0). There are2T(T-1) cones with T independent normals: differences form a tree, whose nullspace is the all-ones vector; adding +/-e0 makes the matrix invertible. Thus each chart cone is simplicial. Interiors partition inputs and class is b. Input ties are measure-zero under the Gaussian; canonical descending order with smaller index first gives deterministic ownership for finite controls.

Keep exact geometric integration distinct from approximate gate freezing. Existing estT halves use the nearest top/second or second/third facet. A single quadratic family has T-1 class-changing facets; choosing one reference facet may change approximate freezing. Exact nearest-facet subdivision has one top-closest piece and T-2 lower-closest pieces per (a,b), hence2T(T-1)^2 chart pieces, cubic. This still removes ordering of lower inputs, but is not quadratic. No full-cost polynomial claim follows from initial family counts.

Freeze before results: (1) independent exact integer/Fraction partition, determinant and reference ordering controls T3..10; (2) Gaussian family mass checks at T3 and old/new exact H4T3 seed0 plus fixed first4 sigma.05 noisy copies seed80; (3) run quadratic family adapter with fixed top facet and exact nearest-facet cubic adapter in both exact and original recipe modes, preserving all outcomes. Use a pinned local estT snapshot, no shared changes; independently flag existing tolerance/tie behavior. No noise chosen by outcome. Count construction separately from inherited operation proxies; import/setup and numerical wall/CPU recorded. <=20CPU review or <=110CPU compute lock,2GB,one thread.

Competing explanations: (A) family invariance is mathematical but larger unsplit cells trigger a numerical topology defect; compare independent polygon exact_acc and mass partition before trusting trained equality. (B) approximate recipe changes because references/cell scales change, despite exact integration invariance; compare both frozen adapters to old recipe and exact references on the same models. A failure does not refute the partition identity.

## Status
Protocol frozen; implementation next.

## Results
None yet.

## Compute accounting
No random model inputs or MC. Weight noise generates fixed model copies for validation only. Shared estT counts are a partial existing convention, not a newly audited complete portable operation count; new construction costs and phase CPU reported separately. No estimator win from family count alone.

## Claims
[inferred] Quadratic ordered top-two cone partition is exact in real arithmetic.
[speculative] It reduces end-to-end exact or recipe cost at unchanged accuracy.

## Alternatives not ruled out
All-facet reference certificates, boundary-only rather than volume integration, compact facet aggregation and higher-dimensional cone mass methods. No factorial lower bound is claimed.

## Failure modes
Boundary ownership, degenerate output ties, floating clipping and nearly singular cone masses may fail independently of the partition theorem. Approximate recipe results need not be invariant under a coarser partition. Dimensions above3 need separately validated mass primitives.
