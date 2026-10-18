# T305: can a closure's per-class error be shared across position classes?

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [T303](../closure_recheck/README.md) · T300 (claude/t218-tie-tail-labels, `r0_shared/`)

## Question
Deterministic closure integrals are too costly per position class (T297, T302, T303). The only lever left is to share
information across classes: compute the closure's error on a few representative classes and reuse it for the
others. How much variance does each position coordinate carry, and does any sharing scheme cut the number of classes
>= 4x with small error?

## Pre-registration (2026-10-18, before any run)
**Data (diagnostic sampling).**
- 10 DEV2 models, every 8th name in DEV2 order (k = 0, 8, ..., 72; both trainers).
- Closures:
  - **R0**: top-3 exact, 7 lows = μ3. Classes = ordered position triples (p1, p2, p3): 720.
  - **T4S** (top-4 exact + spread lows, T298/T303 definition): classes = ordered quadruples (p1..p4): 5,040.
- Per class, K skeletons with those positions and exact-law values (Rényi transform of uniforms), giving the class's
  closure error rate r_c (closure verdict ≠ true 2nd position). K = 2,000 for R0 and K = 400 for T4S.
- **Two independent replicas** A, B (seeds 305000+ and 305500+).
- Each class has exact mass 1/720 or 1/5040, since positions are uniform.

**Sharing error of a scheme** (classes → groups, each with a fixed representative class rep(G)):
- e_m = Σ_G Σ_{c∈G} m_c (r_rep(G)^A − r_c^B), in pp of accuracy. This is the error of computing only the
  representative (replica A) and reusing it for its group, measured against independent class rates (replica B).
- The **noise floor** is the identity scheme (every class its own representative: A vs B).
- Report RMS over the 10 models and the class reduction factor (#classes / #groups).

**ANOVA:** the share of class-rate variance (replica-averaged) explained by main effects p1, p2, p3 (, p4) and by the
pairwise interactions (sequential least squares on the incomplete ordered design). Reported both raw and with the
replica noise variance subtracted.

**Schemes (fixed now):**
1. time shift: key = position differences (p2−p1, p3−p1, ...); rep = earliest member (smallest p1, "anchored at
   start").
2. time shift, rep = latest member (largest p1, "anchored at the end", i.e. ages measured from the end).
3. share over the last coordinate: key = all but p3 (R0) or p4 (T4S); rep = the smallest value of the dropped
   coordinate.
4. share over p1: key = (p2, p3[, p4]) (T300's failing direction, as a control).
5. capped ages from the end: key = min(9 − p_i, A) per coordinate, A = 3; rep = lexicographically first.
6. capped positions from the start: key = min(p_i, A) per coordinate, A = 3 (the start-anchored counterpart of 5).

**Usefulness (DRI):** a scheme is useful if it cuts classes >= 4x **and** has sharing RMS <= .25 pp, both for the
closure in question. Report the best scheme per closure.

**Alternatives first.**
1. As T300 found for p1, the max's age matters, so any scheme that merges classes with different p1 ages fails.
2. Shift invariance fails near the end of the sequence (recent inputs are processed differently), so the end-anchored
   representative should be better than the start-anchored one.
3. Class-rate noise could dominate the sharing error: the identity scheme measures this.

## Status
**Done. No pre-registered scheme is useful** (>= 4x with RMS <= .25 pp). The closest is T4S shared over p4: 7x,
RMS .313 pp. All claims unverified.

## Results
Reproduce: `python3 share.py R0`; `python3 share.py T4S` (about 15 CPU-min in total; per-model rows in `results/*.json`,
summaries in `logs/*.log`). Models: fused_s0, fused_s8, fused_s16, fused_s24, fused_s32, torch_s0, torch_s8, ...
(every 8th in DEV2 order).

### Sharing schemes (RMS over the 10 models of the error of extrapolating a group from its representative, pp)
| scheme | R0: groups (reduction) | R0 RMS / mean | T4S: groups (reduction) | T4S RMS / mean |
|---|---|---|---|---|
| identity (noise floor) | 720 (1x) | .029 / −.000 | 5,040 (1x) | .022 / −.008 |
| time shift, start-anchored rep | 216 (3.3x) | 1.33 / −1.04 | 2,016 (2.5x) | .84 / −.62 |
| time shift, end-anchored rep | 216 (3.3x) | 3.96 / +3.16 | 2,016 (2.5x) | 1.87 / +1.45 |
| share over the last coordinate | 90 (8x) | **.69** / −.16 | 720 (7x) | **.313** / −.04 |
| share over p1 (T300 control) | 90 (8x) | 1.51 / −.78 | 720 (7x) | 1.35 / −.79 |
| capped ages from the end (A=3) | 34 (21x) | 2.25 / −.41 | 73 (69x) | 1.82 / −.21 |
| capped positions from the start (A=3) | 34 (21x) | 2.59 / −1.75 | 73 (69x) | 2.64 / −1.99 |

### ANOVA of class error rates (mean over models, share of the variance; signal share = 1 − replica noise)
| closure | p1 | p2 | p3 | p4 | all main effects | main + pairwise | signal share |
|---|---|---|---|---|---|---|---|
| R0 | .083 | .044 | .044 | – | .182 | .508 | .997 |
| T4S | .118 | .040 | .012 | .003 | .180 | .462 | .982 |

Readings:
- **Class rates are almost all signal** (noise floor .02–.03 pp). They are poorly described by low-order structure:
  main effects explain 18%, and pairwise interactions bring that to only 46–51%. The rest is three-way and higher.
  The age of the max (p1) is the largest single factor, matching T300.
- **Time shift fails** at both anchors (alternative 2 refuted as stated). The end-anchored representative is
  *worse*: shifting a pattern so it ends late changes the error most. The class error depends on the absolute
  positions, not only on their differences.
- **Sharing over the last coordinate is the best direction.** In T4S, p4 carries only .3% of the variance alone.
  Sharing over it returns T4S to 720 triple classes at RMS .31 pp: close to, but above, the .25 bar. Its error is
  a lumpy per-model offset, not noise (noise .02).
- **Capped ages/positions** (21–69x) fail by 1.8–2.6 pp.

## Claims
- S1 [measured] No pre-registered scheme cuts classes >= 4x with sharing RMS <= .25 pp, for R0 or T4S.
  Best: T4S over p4, 7x at .313 pp; R0 over p3, 8x at .69 pp.
- S2 [measured] The per-class closure error rates have strong high-order position interactions. Main effects explain
  18% of the variance, + pairs 46–51%, with more than 98% signal. That is why one-representative sharing fails.
- S3 [inferred] Combined with T297/T303: T4S shared over p4 would need >= 720 × 2 = 1,440 FPE for the floor design
  (one verdict per triple and orientation) with no boundary resolution, at an extra .31 pp sharing error. Resolving
  the boundary per triple brings back T297's >= 20k FPE. So class sharing does not open the cheap regime for these
  closures.

## Alternatives not ruled out
- Representatives chosen differently (the median p4 instead of the smallest, or 2 representatives averaged, which
  halves the reduction) were not tried. They might bring T4S over p4 under .25 pp, but at <= 3.5x.
- Fitting an interaction model on a computed subset of classes is a regression, so it would be inductive. Not tried.
- 10 models. The scheme ranking was consistent across them, but individual errors vary (e.g. R0 shift_end
  1.0–8.9 pp).

## Compute accounting
Per model: R0 2 × 720 × 2,000 = 2.9M forwards; T4S 2 × 5,040 × 400 × 2 = 8.1M forwards. That is 1.1e8 forwards for
10 models, about 15 CPU-min. Diagnostic sampling.
