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

### First frozen results and retained defect
Commands: single-thread review lock and `ulimit -v 2500000; ulimit -t20; /workspace/clean-neurons/.venv/bin/python m16/approaches/class_region_families/check.py` (2.788CPU), then `ties.py`(.280CPU). Matrix integer-inverse and constructed-interior controls T3..10 pass; exhaustive distinct permutations T3..6 pass; all1080 tied ternary input ownership controls T3..6 pass. T3 family masses sum to1 with1/3 perclass. Both exact adapters match independent spherical-polygon exact_acc on allfive fixedmodels within3e-14. Quadratic exact inherited+construction proxy is65–69% of old; this is not an independently audited full operation count. Approximate quadratic changes four of five values and worsens their absolute errors, whereas cubic nearest reproduces oldrecipe to floatingprecision. Thus approximate invariance is false even though exact invariance holds.

Preserved failure: zero or identical readout rows with nonzero hidden states produce.99833013148 rather than1/3 in every partition. Existing engine `_cell_error` assigns equality to the candidate truth even when a lower output index wins an identically-zero margin. This is an inherited output-tie defect, independent of the class partition. Before repair, freeze a local adapter: when Ltruth-Lo is identically zero, return the whole remaining candidate cone as error iff o<truth; otherwise skip the comparison. Near-zero tolerances unchanged. Re-run original fixtures and trained models unchanged, plus duplicate-output positive/negative margin witnesses. Shared source remains untouched; root/DRI notified.

Alternative proposed independently by root: sign(sum x) chart makes Gaussian mean independent of tree-difference coordinates, so each signed ordered-top-two family has mass1/[2T(T-1)]. This may simplify checks, but loses the initial-gate sign guarantee and is not substituted into frozen experiments.
