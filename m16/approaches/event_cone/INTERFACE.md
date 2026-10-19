# Certified rare-event bound lane (T398 lead, T399 GPU bounds, T400 rigour): interfaces v1 (2026-10-19)

↑ [README](README.md). Owner: T398 lead (session_01LhvQ3Abu3KY6KE3AELyQUg). Branch `claude/t373-event-cone`, folder
`m16/approaches/event_cone/`. Helpers push only into their own sub-folders here: `gpu_t399/` and `rigour_t400/`.

## Model and event
- M_16,10 = `alg_zoo.example_2nd_argmax()` (same weights as `m16/common.py` Wi, Wh, Wo; float64). There are no biases.
  - h_0 = 0; h_t = relu(Wh h_{t-1} + Wi x_{t-1}) for t = 1..10, reading position t-1 at step t; logits L = Wo h_10.
- x ~ N(0, I_10). Triple (p1, p2, p3): 720 ordered distinct positions.
- **Event E(t0)** = { x_p1 >= x_p2 >= x_p3 + t0, x_p3 >= x_l for every low l, argmax L = p3 }, i.e. T340's E1: the
  network outputs the 3rd-largest position while the 2-3 gap exceeds t0.
- **Goal:** a rigorous UPPER bound on P(E(t0)) = sum over the 720 triples of P(E restricted to the triple's cone).

## Node (one per prefix cell, plus optional ReLU splits)
| field | shape | meaning |
|---|---|---|
| `tri` | (3,) int | (p1, p2, p3) |
| `k` | int | steps already exact (0..10) |
| `G` | (m, 10) float64 | homogeneous half-spaces G x >= 0: the triple's ordering cone plus the gate splits of steps 1..k. m <= ~120 |
| `A` | (16, 10) float64 | h_k = A x exactly on the cell (A = 0 at k = 0) |
| `phase` | (10-k, 16) int8 | splits on remaining steps: +1 active (z >= 0, h = z), -1 inactive (z <= 0, h = 0), 0 free |
| `hist` | bytes | gate history of steps 1..k (np.packbits per step), the node id |

Normalisation (both bound and mass calls): the network and the cone are scale-invariant, so infeasibility is decided
on P = { G x >= 0, |x_i| <= 1, x_p2 - x_p3 >= DELTA }, with DELTA = 0.1. Event points with g23 < DELTA ||x||_inf have
||x||_inf > t0/DELTA, mass < 20 Phi(-t0/DELTA) (< 1e-49 at t0 = 1.5), which is charged once globally.

## Bound call (T399, GPU): `prune(nodes) -> results`
- Per node, decide whether **no x in P has L_p3 >= L_j for all j != p3**.
- Sufficient test: find lambda in the simplex over j != p3 with min over the relaxed node of sum_j lambda_j (L_j - L_p3)
  > 0 (lambda optimised like alpha in alpha-CROWN). Equivalently, an infeasible LP relaxation.
- Return per node: `status` (1 pruned / 0 unknown), `margin_lb` (the best lower bound of max_j (L_j - L_p3); > 0
  means pruned), and the certificate (below).
- Reference implementation (CPU LP, the ground truth for tightness and soundness checks): `relax.py`
  (T390, triangle with forward LP bounds) and `relax2.py` (T398: + event-aware tightening + splitting), plus
  `relax2.max_margin` (relaxation upper bound of max_x (L_p3 - max_j L_j); the negated quantity).
- Benchmark nodes: `nodes/bench_182_k4.npz` (all 273 survivors of (1,8,2) at k = 4 under T390, with
  their T390 LP max-margin, the T398 event-tightened value, and a hit-and-run best-found margin). Format: arrays
  `G` (N, mmax, 10) zero-padded + `m` (N,), `A` (N, 16, 10), `k`, `tri`, `relax_margin`, `tight_margin`,
  `probe_best`, `is_event` (1 if a line-walk event point lies in it).

## Certificate (consumed by T400)
For a pruned node, an npz with the node fields, and:
- `lo, hi` (10-k, 16): the pre-activation bounds used;
- for each bound, its proof: dual multipliers y >= 0 of its bounding LP, or CROWN linear coefficients plus the
  concretisation multipliers over P, such that weak duality gives the bound;
- the final proof: a Farkas ray (LP relaxation infeasible), or the lambda vector plus its CROWN coefficients.

T400 re-derives every relaxation row from (G, A, Wh, Wi, Wo, lo, hi, phase) and checks each inequality in exact
rational (or outward-rounded interval) arithmetic. T400 owns the exact format; this is the content.

## Mass call (T400): `mass_ub(node, t0) -> float`
- A rigorous upper bound on N(0, I)( { G x >= 0, x_p2 - x_p3 >= t0 } ), i.e. the cell restricted to the far-gap
  slab (event ignored).
- At k = 10, an optional tighter call `event_mass_ub(node, t0)` adds the 9 homogeneous rows L_p3 - L_j >= 0
  (exact on a k = 10 cell, since h_10 = A x).
- Use: **charge and stop**. A node whose mass_ub is below a per-node allowance is charged and not expanded. Leaves at
  k = 10 are charged event_mass_ub.
- The certified result is
  U = sum(charged) + 20 Phi(-t0/DELTA) + float slack,
  with every pruned node carrying a verified certificate.

## Target
t0 is chosen so that the truth is < ~1e-9 (README "Target threshold"). The truth comes from T340's conditional-law
estimator along x_p3 (exact 1-D law, unbiased), for comparison only.
