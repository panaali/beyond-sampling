# T408: deductive estimator for the depth-1 MLP D1w128 by conditioning on the top-3 positions

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [T345 ffn_control](../ffn_control/README.md)

Owner: Claude child session_01Na5o6K4TXWJmNtj7N5f76X (T345 child), branch `claude/t345-ffn-control` (the session's
designated branch; a separate folder instead of a separate branch).

## Question
T345 found that the simplest all-inputs-at-once net reaching .95 (D1w128: one ReLU layer of 128, bias-free) is
built from near-exact pairwise comparators relu(x_a - x_b) (76-79 units) and "both low" detectors (49-52 units).
Can a deterministic estimator (no random inputs) use that structure? The T408 plan: condition on which 3
positions hold the 3 largest inputs and in what order (720 cases). Every comparator touching a top-3 position then
has a fixed gate, so the hope was that the top-3 margins become linear in the 3 top values plus ~2 projections of
the 7 lower values. P(correct | case) would then be a ~5-D truncated-Gaussian integral done by deterministic
quadrature, with three corrections bounded or computed:
- (i) both-low gates left undetermined by the case;
- (ii) comparator gate errors near ties;
- (iii) winners outside the top 3.

## Method
Design diagnostics before any pre-registered scoring (`python diagnostics.py`, results/diagnostics.json). They
are sampled, on D1w128 2^30 seed 0 (a member of the T345 D1w128@30 panel), and look only at the net's structure,
never at panel truths.
1. Where the errors go, and how accurate the gates fixed by a case are.
2. How large the contribution of the case-undetermined units is, in margin units.
3. The decisive test. Condition on the ordering of the top k inputs, k = 3..10 (k = 10 is a single ordering
   cone). Fit by least squares the best linear function of x to the two margins z_2nd - z_1st and z_2nd - z_3rd,
   then compare the error rate this linear surrogate implies with the true conditional error rate. A deductive
   rule can do no better than this sample-fitted linear surrogate, so if the surrogate fails, any route that
   treats the case as linear fails.

## Status
Done (2026-10-19): **scoped negative at the design stage**. No scored estimator was built, because correction (i)
does not close. The pre-registration the DRI asked for (frozen quadrature rule, correction bounds, FPE accounting)
was therefore not written. Its premise, near-linear margins per case, fails the test above.

## Results
`python diagnostics.py` (D1w128 2^30 seed 0, accuracy .990).

| quantity | value |
|---|---|
| error rate (4e5 samples) | .0105 (predicts the max .0036, the 3rd .0069, outside the top 3 .00002) |
| comparators fixed by a case / their gate error | 42.1 per case / .026% |
| both-low units with a top-3 input / fraction on | 26.1 per case / **21%** (not "off" as hoped) |
| units undetermined by a case | **59.7** per case |
| top-3 margins, median abs | 13.5 (2nd vs 1st), 9.9 (2nd vs 3rd) |
| contribution of undetermined units to those margins, sd | **41.9, 41.8** |

Conditioning on the top-k ordering, with a least-squares linear fit of the margins in x (one case prefix):

| k | cases | residual sd (m12, m23) | P(err given case) | P(err) of linear surrogate |
|---|---|---|---|---|
| 3 | 720 | 3.24, 2.61 | .033 | .072 |
| 4 | 5,040 | 3.04, 2.51 | .033 | .073 |
| 5 | 30,240 | 2.80, 2.38 | .032 | .078 |
| 6 | 151,200 | 2.51, 2.37 | .032 | .087 |
| 7 | 604,800 | 2.39, 2.33 | .031 | .085 |
| 8 | 1,814,400 | 2.32, 2.21 | .030 | .088 |
| 10 (one cone) | 3,628,800 | 2.31, 2.06 | .031 | .092 |

(This case prefix is harder than average. Three other top-3 cases checked by plain MC on 4e6 samples give .023-.036
against .0104 overall, so the sampler is consistent.)

- **Correction (iii) closes**: winners outside the top 3 carry 2e-5 of mass.
- **Correction (ii) closes**: fixed comparator gates are wrong .026% of the time.
- **Correction (i) does not close.**
  - About 60 units per case are undetermined: the 21 both-bottom comparator pairs, and both-low units, including
    those with a top-3 input, which are on 21% of the time.
  - Their contribution to the decisive margins has sd ~42, against margins of 10-14. So the bottom 7 inputs do
    not enter through ~2 projections.
  - Even the best sample-fitted linear surrogate leaves a residual sd of 2-3 margin units and over-predicts the
    conditional error rate 2.2-3x.
  - Conditioning on more of the ordering barely helps. Even a single ordering cone (k = 10) leaves the both-low
    gates nonlinear exactly where decisions are made.
  - Treating those gates exactly means enumerating the ~1.3k+ gate subregions per cone (T345 cone_test), i.e. the
    region route already found dead.

## Compute accounting
Diagnostics only: ~2 CPU-minutes. No estimator was run, so no FPE cost or F is reported.

## Claims
- [measured] On D1w128 (2^30, seed 0), top-k ordering conditioning leaves margins whose best linear surrogate gets
  the conditional error rate wrong by 2.2-3x for every k from 3 to 10; correction (i) does not close.
- [measured] Corrections (ii) and (iii) are small (.026% gate error; 2e-5 outside-top-3 mass).
- [inferred] No deterministic estimator that treats each top-k case as linear can reach the strict bar (F <= 1
  needs RMS error ~.1 pp at C ~ 1e4 FPE, i.e. ~10% of the error mass), nor the lenient bar at the error levels
  above (a 2-3x error in the error mass is ~1-2 pp).

## Alternatives not ruled out
- Treat the both-low units exactly but with an aggregate. Each depends on x only through (x_a + x_b) against a
  weighted mean. A route that integrates over these ~50 low-dimensional statistics jointly (rather than cone by
  cone) is untested; its dimension is high, though.
- A nonlinear per-case surrogate (e.g. quadratic, or piecewise in one or two dominant both-low gates) derived from
  the weights: untested.
- The diagnostics use one model (seed 0) and one case prefix for the top-k table (three more cases for the error
  rates). Other seeds have the same unit structure (T345 mechanism table) but were not run here.
- The linear surrogate was fit to all inputs of the case. Fitting near the decision boundary only would do
  better near ties, but a deductive rule cannot fit at all, so this is a bound on the idea, not on every
  estimator.

## Failure modes
- Sampled diagnostics: 3e5-4e5 samples per estimate (SE of the error rates ~.0003).
- The comparator / both-low classification uses the sign-checked rule of T345 (after T394).
