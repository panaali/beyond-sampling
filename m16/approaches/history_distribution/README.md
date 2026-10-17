# T76 positive conditional hidden-history distributions

## Question
Does preserving within-sign hidden-state variation repair the two-mean history surrogate, or is the missing dependence between hidden state and prefix ranks still dominant? TASKS-only0043e75 was published and ownership rechecked before implementation. Own only this folder; prior nonzero_history and rank-conditioned engines stay unchanged.

## Method
For T4, the omitted prefix has exactly two iid standard normal inputs. Conditional on sign s of X0, write X0=s R cos(theta), X1=R sin(theta), where theta is uniform on[-pi/2,pi/2] and R is independent Rayleigh. The two-step state is R ReLU(c cos(theta)+Wi sin(theta)), c=Wh ReLU(s Wi). Split theta at every gate zero. On each gate sector subdivide into2 or4 equal panels, and replace its direction by its exact angular mean. This keeps the conditional hidden mean exactly and produces positive states on the correct origin/axis/interior support. It loses a computable positive-semidefinite within-panel covariance, which must be reported.

Represent R by positive Gaussian quadrature with2 or4 nodes, constructed from exact-form moments E[R^k]=2^(k/2)Gamma(1+k/2). A Hankel Cholesky/Jacobi eigenproblem supplies nodes/weights; control all required moments, positivity and conditioning. The first two moments are exact in real arithmetic for both rules. All setup is charged. No conditional Gaussian with negative hidden-state support is used.

For each sign, average the existing exact affine two-suffix rank calculation over these positive hidden atoms. Crucially, hidden atoms are independent of the actual omitted-prefix rank variables conditional on sign. Thus this is a deliberately decorrelated surrogate: it isolates state variation while retaining the previous first-sign rank law. It is NOT an exact estimator of the original network merely because the marginal state law improves. Constant-state collapse must reproduce the old exact-mean surrogate.

## Frozen protocol and alternatives
Before trained scoring, check Rayleigh moments through2n-1, positive nodes/weights, conditional state mass/mean against the analytic formula, analytic second moments and PSD covariance loss, origin/axis masses, zero-Wi/Wh0 cases and positive diagonal coordinate scaling. Every source-cell mean must stay nonnegative without clipping negative states. Verify constant atom collapse reproduces previous mean estimates and constant readouts reproduce rank controls.

Freeze the2x2 factorial angular subdivisions2/4 and radial nodes2/4 on exactly the prior H2T4 base and relative .02 copy0 failure. Both are prior T76 cases, no new noise or inputs. Production suffix GL16, diagnostic GL32 on identical atoms; preserve every output and cost, no selected winner. The higher rule is a consistency diagnostic, not truth. References enter scoring only. <=110CPU per cycle, one thread/2GB/shared locks; short controls<=20CPU review lock. No larger history horizon, new model or sweep automatically follows.

Alternative1: state variation was the dominant lost information; positive mixtures improve the failed noisy case and remain stable under the factorial checks. Alternative2: state/rank coupling dominates, so better marginal state variation fails or converges to the wrong surrogate accuracy. Alternative3: angular/radial approximation or suffix integration dominates; exact mean, covariance/mass controls and independent rule refinement distinguish this. A failed fixed mixture does not reject all history models.

## Status
Protocol frozen before implementation; independent T55 algebra review requested.

## Results
Pending. Preserve all errors and numerical failures.

## Compute accounting
Count gate-root discovery, angular panels, radial moment/Hankel/eigen setup, atom construction, every affine arrangement and scalar-rank integration, and cold CPU. Existing event counters remain partial, not portable FLOPs. Source/model hashes and prior references are recorded. No random model inputs or model-fitted constants enter this construction.

## Claims
[inferred] The conditional polar representation is exact; [inferred] panel-average directions preserve conditional means and support, with PSD covariance loss. [speculative] Adding the corresponding positive state mixture repairs the mean-surrogate error. Only finite independently checked results can support that claim.

## Alternatives not ruled out
Explicit rank-conditioned state coupling, more history signs, non-tensor reachable distributions and a different target-last solver remain open and separately owned. No broad impossibility or sampling win is claimed.

## Failure modes
Rounded coincident gate angles, tiny angular panels, ill-conditioned moment matrices, steep atom-dependent suffix accuracy, missing state/rank coupling and component-count growth. No epsilon merging, negative-state clipping or normalization fitted from accuracy is allowed.

## Primitive control checkpoint
All16 distribution fixtures (four gate/support cases times angular2/4 and radial2/4) and three model controls pass in1.372302479CPU. Positive diagonal scaling commutes with atom construction to the frozen tolerance; conditional means/masses and Rayleigh moments pass; covariance loss is PSD. Collapsing atoms to their mean reproduces the old sign-mean surrogate, and zero-readout/Wh0 controls reproduce previous rank results. Exact duplicate states are coalesced within sign without any tolerance; this preserves origin mass and avoids identical arrangements. Source dictionary/matrix-library overhead remains named, not a portable arithmetic certificate. Proceed only to the frozen two-case factorial.
