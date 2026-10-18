# r1_b2b5 (T307): R1W V1 + B2 (exact prefix sharing) + B5 (exact last-input marginalisation)

↑ [r1_walk (T304/T306)](../r1_walk/README.md) · Fable round-3 review B2, B5 (branch claude/project-handoff-priorities-zpzfm8,
`review/fable_review_r3/README.md`) · [TASKS](../../../TASKS.md)

**Scope: lenient bar, retrains. The claim is HP2 only. DEV2 is development data.**

## Pre-registration (2026-10-18, before any run)
**B5 adapted to R1W's closure.** In closure c, v4 sits at the latest free position, which is position 9 whenever 9 is
not in the triple. So x9 is **never a low**: it is either a top-3 value (exact) or v4. B5's literal form ("x9 is a
low") is vacuous here. The adaptation:
- where x9 = v4 (70% of triples), replace v4's 1-node point value by its **exact conditional law** (max of 7
  N(0,1) given < v3), pushed through the last step only;
- the 6 lows stay at mu(v4_node) (declared, as in V1);
- the verdict becomes P(correct) over x9.

**(1) B5 oracle check (diagnostic sampling), 10 DEV2 models (T297 set), 100k skeletons each, paired.** Closures:
- (i) **c-node** = V1's closure: v4 at its conditional mean given v3, at the latest free position;
- (ii) **c-B5**: x9 = v4 by its exact law (dense 1-D quadrature in this diagnostic), with lows at mu(v4_node);
- (iii) **c-true**: v4's true value (the .471 reference).

Floor = RMS over models of (closure accuracy - truth). **Stop** if floor(ii) > 0.9 x floor(i), i.e. a reduction
< 10%.

**(2) V2 = V1 + B2 + B5, nothing else changed.**
- **B2 changes cost only.** The first cell of every walked line is charged by distinct (input, tangent) prefixes
  across the 720 triples: sum over steps s of #unique prefixes x 2/T FPE. Later cells are charged as now (restart
  from the flipping step).
- B5 inside the walk is implemented at the level that keeps the walk exact (design fixed after (1), before any V2
  scoring).

**(3)** Freeze V2 and send it to the T281 scorer (HP2). Report DEV2 F as development.

**Forking paths:** this is the **third HP2 submission** in this family (after T300 stage 0 and T306 V1).
