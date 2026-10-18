# r1_skip (T311): cut R1W V1's cost by skipping walks that provably cannot flip — feasibility killed before building

↑ [r1_walk (T304/T306)](../r1_walk/README.md) · [r1_b2b5 (T307)](../r1_b2b5/README.md) · [TASKS](../../../TASKS.md)

## Question
At HP2's RMS of .99 pp, the lenient bar needs C <= ~8.1k FPE, against V1's 13.3k. Can a deductive per-line test
("no decision flip within the walk range") skip enough walks?

## Feasibility checks (2026-10-18; diagnostics only, before any skip rule was fixed or anything was scored)
**(1) Upper bound on the saving** (`fused_s8`, V1's lines, tmax .5). On a line with no decision flip, the correct set
is the whole range or empty.

| line | no-flip fraction | walk FPE | saveable beyond the first cell |
|---|---|---|---|
| nonblind g23 | .48 | 4,330 | 1,414 |
| nonblind g12 | .50 | 3,549 | 1,052 |
| blind g23 | .96 | 2,364 | 887 |
| **all** | **.65** | **10,243** | **3,353** |

- **Even a free and perfect certificate saves at most 3.35k FPE**: 13.7k down to ~10.4k. The first cell (value +
  tangent at the tie) is needed to start any test.
- Adding B2's exact first-cell sharing (T307: 4,320 -> 1,689 FPE) gives a best case of **~7.8k FPE**, just under the
  8.1k target, and only if the certificate costs nothing and misses nothing.

**(2) The only free certificate: a weight-derived Lipschitz bound** on the margin along the line, using the margin and
tangent from the first cell. Gate-agnostic:
|dm/dt| <= max_ij ||Wo_i - Wo_j|| x ||Wi|| x sum_t ||Wh||_2^(9-t) |d_t|.

| model | ||Wh||_2 | Lipschitz bound (all steps vary) | (only the last step varies) |
|---|---|---|---|
| fused_s8 | 4.32 | 2.3e8 | 336 |
| torch_s0 | 5.09 | 1.2e9 | 426 |

- Margins are O(1-10), so the bound certifies a flip-free stretch of ~1e-8. Even with only x9 varying it certifies
  ~0.01 of the 0.5 walk. **It certifies essentially nothing.**
- A certificate tight enough to help is linear-relaxation (CROWN/IBP-style) bounding of the margin over the segment.
  That costs ~2-3 FPE per line, more than the ~1.6 FPE per line that perfect skipping saves on average
  (3,353 / 2,160 lines).

## Verdict
**Killed by arithmetic before building.**
- A free deductive certificate is ~1e8x too loose.
- A useful one costs more than it saves.
- Even a perfect free one only just reaches the 8.1k target with B2.

So no skip rule was pre-registered or scored, and **no fourth HP2 submission was made**. The forking-paths count stays
at two HP2 submissions in this family (T300 stage 0, T306 V1).

## Alternatives not ruled out
- Gate-aware local bounds that reuse the first cell's gate pattern (exact within the cell) plus a cheap bound on the
  number of later flips. Not derived; the fragmentation near ties (H-E15) makes it unlikely to certify much.
- Cutting cost elsewhere: the blind g23 line (.96 no-flip, 2.4k FPE) could be dropped as a declared approximation
  instead of being certified. That is an accuracy-for-cost trade with its own error, and would need its own
  pre-registration.

## Compute accounting
Two diagnostics on one or two models: seconds.
