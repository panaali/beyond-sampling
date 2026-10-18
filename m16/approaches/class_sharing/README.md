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
Pre-registered.
