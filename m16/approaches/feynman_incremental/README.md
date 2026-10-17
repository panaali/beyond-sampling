# T20: incremental Feynman estimator port

## Question
Can the existing sci_feynman quadrature retain its face law while replacing historical recurrence-only charges and unsafe legacy walking with the corrected incremental walker? TASKS-only claim b0950a0 was pushed from a fresh administrative checkout before implementation. This folder owns the port; T30 owns the independent legacy-walker audit. No original/shared sources are edited.

## Method
Freeze the first phase: adapt the pinned sci_tao_v2 Python walker to sci_feynman's R23/R12 target convention and excluded center. For0<=tex<tau, wrong mass on tex<=|t|<=tau equals full-window mass at tau minus full-window mass at tex. R23 uses kind23 and R12/top uses kind12. Empty windows return0; tex=0 skips the second call. Charge every executed window call, prefix setup, local subtraction and validation; this deliberately recomputes overlapping windows instead of claiming unimplemented reuse. Return explicit per-call counters. Refuse invalid/nonfinite windows and preserve available partial work on failure; never return truncated mass as a completed estimate.

The initial adapter is a numerical integration primitive, not a full estimator. After it passes, retain the original model-free configuration enumeration and node schedule in a local estimator adapter, then inventory and charge node generation, transforms, blind evaluation, recurrence, readout and aggregation. Precomputation, opaque library work and cold import/setup must be named rather than called free. Publish est(model,budget,rng) only with its actual cost convention and limits; historical estimator scores do not transfer to this port.

## Frozen alternatives and falsifiers
1. Subtracting two complete intervals may have the wrong target sign or exclusion orientation. Compare exact binary-rational gate/readout reference intervals from independent T30 on R23 and R12, tex in{0,.0625,.25,.5}, including empty windows and positive-volume ties.
2. The corrected shared walker may still fail near roots or the adapter may mask cap exhaustion. Reuse T30's9 predeclared fixtures, particularly its close affine envelope crossings, and force cap0/1. Errors must propagate with available partial costs; no fallback success value.
3. Independent full-window evaluation may lose precision by cancellation when the annulus is tiny. Freeze tex=nextafter(tau,0) cases; report any negative/poor relative result rather than applying an arbitrary scale-dependent tie threshold. An interval-native engine is a later alternative if this simple adapter fails.
4. Old/new whole-estimator disagreement may reflect old bugs, changed nodes or cost conventions. Later comparisons must retain exact node schedules/source hashes and classify each discrepancy before changing geometry or interpreting accuracy.

## Compute accounting
One thread,2GB RSS, batches<=65,536, primary lock for longer checks; review lock only for<=20CPU primitive checks. Each phase<=110CPU. Shared walker counters use its declared FPE/scalar-operation proxy, including CDF50, not a machine-instruction certificate. Every local validation/combination operation is additionally charged. Whole-estimator input-generation/precomputation costs are pending the second phase and cannot yet support an end-to-end sampling comparison. Independent T30 exact-rational/reference checks are separate diagnostic cost.

## Claims
- [measured] No new port experiment has run at this protocol commit.
- [inferred] Full-window subtraction gives the same excluded-center integral in exact arithmetic, with both target conventions explicit.
- [speculative] It may be slower than an interval-native walker but is a bounded, reviewable first port with honest recomputation.

## Alternatives not ruled out
Interval-native integration, merging the two walks, compiling the validated interface, and more efficient model-free node generation remain open. This phase does not port tube, Laplace, boundary or control-variate estimators; those remain T20 follow-ups. A numerical adapter failure does not refute the underlying quadrature method.

## Failure modes
Cancellation in very thin annuli, inherited floating root probes, cap exhaustion, invalid model/window inputs, incomplete partial counters after numerical exceptions, hidden node-generation costs and budget overshoot. No silent clipping or default estimate on failure. Claude remains DRI; independent validation/classification is separate from implementation.

## First adapter cycle
[measured] The9 independent T30 exact-rational fixtures, both target conventions and four ordinary exclusions produce72 passing cases. All18 adjacent-float thin-annulus controls also satisfy the frozen absolute tolerance; maximum absolute difference over90 cases is1.943e-16. There are no negative-mass or other adapter failures. CPU0.244 seconds, no random model inputs. Empty windows execute no shared walk; nonempty tex=0 windows execute one. Full/excluded calls and adapter work are saved separately in `adapter_results.json`.

[measured, precision limitation] All18 thin-annulus true masses are positive but the subtraction returns0, giving relative error1 despite tiny absolute error. This is preserved as a failed relative-accuracy alternative, not hidden by clipping or called an exact result. An interval-native engine with stable CDF differences is a potential rescue; ordinary-window agreement does not resolve this relative limitation.

[measured] Forced gate-at-zero cap0/1 failures propagate explicitly, retaining358/992 known operations respectively, including48/682 shared-walker operations already spent. No truncated mass is returned. For an unexpected numerical exception the ledger marks unresolved failed-kernel cost rather than inventing a complete charge. Shape/model validation and whole-estimator generation/setup accounting remain subject to the next bounded audit; this primitive is not an end-to-end cost-complete estimator.

## Frozen interval-native rescue
Keep the subtraction adapter/results intact. Build a separate local function from the pinned corrected walker, changing only the interval endpoint layer: intersect negative/positive target intervals with(-infinity,-tex] and[tex,infinity), then add the Gaussian masses of the two wrong complement pieces directly. Continue walking all gates, including the excluded center, and charge that work. No new gate algorithm, original-source edit or threshold-based argmax rule.

For a finite interval of width delta and midpoint m, midpoint mass is delta*phi(m). Its real-arithmetic absolute quadrature error is at most delta^3/(24*sqrt(2*pi)), since sup|phi''|=1/sqrt(2*pi). Use this positive formula only when that bound is no larger than machine epsilon times the computed midpoint mass; otherwise use the ordinary CDF difference and retain its existing floating qualification. Zero-width intervals return0. This mathematical selection affects mass evaluation only, never tie ownership or root selection. Integrating complement pieces directly avoids subtraction of nearly equal full/correct masses.

Charge a conservative220 scalar-equivalent operations per attempted normal interval (including two CDF50 and EXP50 plus ordinary arithmetic) and40 per output assembly, in addition to the unchanged shared ledger; obsolete shared CDF allowances remain conservatively duplicated. Reuse the exact90 cases and cap fixtures, require thin-annulus relative agreement where the independent true mass is positive, and preserve every failure. No new trained-model sweep. Independent reviewer should check the midpoint derivative bound and local-source replacements before full-estimator adoption.

T55's independent algebra review confirms the cubic bound and flags that an adjacent-float interval's true midpoint may not be representable; a rounded midpoint would add an order-delta-squared center-shift term. Before implementation, choose the exact algebraic midpoint expression `delta/sqrt(2*pi)*exp(-a*a/2-a*delta/2-delta*delta/8)` instead of explicitly forming the midpoint. The expression has the cubic quadrature remainder in real arithmetic; ordinary floating arithmetic/exp rounding remains separately qualified. This preserves the single-EXP220-operation allowance.

## Interval-native rescue result
[measured] The separate `native.py` adapter passes the same72 ordinary and18 adjacent-float controls with maximum absolute error1.943e-16. All18 positive thin masses are now retained, with maximum relative error below3e-16, instead of the subtraction adapter's relative error1. No failed case or silently clipped value occurred. CPU0.517 seconds; no trained-model sweep. Forced cap failures still retain known work and return no completed estimate.

[inferred, independently reviewed algebra] T55 review f813e13 validates the midpoint quadrature argument with the algebraic-center correction. The native local source transformation changes only target-half interval clipping and the Gaussian complement integration; the pinned shared gate/cache implementation is unchanged. It walks the full gate interval even across the excluded center and charges that work. The inherited CDF allowance and new700-per-envelope allowance deliberately overlap conservatively. This is an operation-proxy convention, not instruction-level certification.

The native primitive is the candidate for later estimator integration, subject to independent source review. Neither primitive yet includes a cost-complete whole-estimator implementation, node-generation audit, cold geometry-table accounting or a newly scored accuracy-versus-sampling result. The subtraction implementation and its failed thin relative-accuracy check remain preserved.
