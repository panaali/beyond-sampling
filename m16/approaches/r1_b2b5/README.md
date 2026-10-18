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

## Results
### (1) B5 oracle check — **STOP** (`python b5_oracle.py 100000`, `results/b5_oracle.json`; 10 DEV2 models)

| closure | RMS floor (pp) |
|---|---|
| (i) c-node (V1's closure) | 0.503 |
| (ii) c-B5 (x9 = v4 by its exact law) | 0.461 |
| (iii) c-true (true v4 value; .471 on all 80) | 0.519 |

- **The reduction is 8.2%, below the pre-registered 10%. Stop: V2 is not built and nothing is sent to HP2.**
- Correcting my T306 guess: **the v4 point node is not a meaningful error source.** c-node is no worse than c-true
  on these models.
- V1's excess error over its closure floor therefore comes from the walk-level declared approximations: one
  transverse point per region (B tried and worse), the dropped blind g12, the v4-v3 linkage and the mu
  linearisation. It does not come from v4's value.

### B2 cost (measured exactly, `python b2_count.py`; cost only, the estimate is unchanged)
- First cell of every walked line, distinct (input, tangent) prefixes across the 720 triples: **2,815 vs 7,200
  step evaluations (2.56x)**. That takes 4,320 FPE down to 1,689.
- Later cells have no cross-row sharing: rows sit at different walk parameters after the first cell. Restart-step
  accounting already avoids recomputing the unchanged prefix.
- Projected V1 + B2 cost: ~10.7k FPE (HP2 accounting 13.3k - 2.6k). With HP2's RMS of .99 pp, the projected HP2 F
  is ~13.2. Lenient needs <= 8.1k at that RMS, so **B2 alone cannot pass** [inferred projection, not run].

## Status
Done (2026-10-18). Stopped at (1) by the pre-registered rule. No third HP2 submission was made, so the
forking-paths count stays at two HP2 submissions in this family (T300 stage 0, T306 V1).

## Compute accounting
B5 oracle: 10 models x 1e5 skeletons x (2 + 64) forwards ≈ 6.6e7 forwards (diagnostic), ~6 CPU-min. B2 count:
seconds.
