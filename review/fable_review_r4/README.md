# Fable review, round 4 (2026-10-18): the tie-wall flips, and what to do about them

Read (via `git show`, nothing run): shared `ccr-6a61f25c-wn80ii` (INDEX T281, T296-T316; TASKS Directive 8; SCALING_OBSTRUCTIONS §9), `claude/t218-tie-tail-labels` (r0_walk `walk.py` + README, r1_walk README + `r1w.py`, r1_capped, r1_v1b2), r0_variants, info_ceiling2, shatter_test, the Hilton notes, my round-3 review. Child results unverified unless INDEX says otherwise. Arithmetic only. Jargon as in round 3; FPE = 5,948 FLOPs; F = (RMS/.85 pp)^2 x C_eff/1k; lenient F <= 10 (upper CI too), strict F <= 1.

## A. What changed

T308/T314 set the information ceiling (cheap features .72-.86 pp vs the .43 strict needs), T309/T310/T305 killed the non-walk levers, and T315/T311 pinned the R1W lineage between two facts: extrapolating through the first gate cells costs 5-10 pp of bias, and walking them costs ~2 FPE per cell. R1W V1 (F 14.92 [10.74, 19.38] at 12.1k FPE, RMS .99) is reproduced and honestly costed.

**Reading `walk_line` changes the question.** Two things there are facts about the implementation, not the problem:

1. Every later cell is charged (and run) as a value+tangent restart from the flipping step. But a ReLU gate flip is *continuous*: the state at the flip point is the previous cell's affine form evaluated there. Only the tangent changes, by the propagation of one coordinate through the existing gated chain. The value half of every later cell is redundant.
2. The margin's kinks along a line are events (step s, neuron i). To first order their locations are the ratios -z_{s,i}(0)/dz_{s,i}(0) of the first cell's pre-activations to their tangents (already computed), and their effect on the margin's slope is lambda_{s,i} x dz_{s,i}, with lambda the adjoint (one backward pass). Nobody in the R0 lineage has run an adjoint.

The first gives an exact cost cut with identical estimates; the second gives the cheap approximate walk the brief's (a) and (b) ask for. Neither is on the board.

## B. Three designs

### D1. Exact walk by continuity (cost only; estimates identical to V1)

*Computation.* First cell as now (value + tangent, shared by the B2 prefix tree, which T315 showed is exact: 7,200 -> 2,815 step-evaluations). At a flip (s, i) at t*: take z(t*) for all (t, j) from the current cell's affine form (continuous); set the new gate; recompute only the tangent from step s, dh_t = D_t (W_hh dh_{t-1} + W_ih d_t), ~560 FLOPs per step plus ~80 for the roots; no value pass. Readout and correct-set solve as now, so the estimates match V1 to rounding.

*Cost.* A later cell now costs 2(T-s)/T FPE, ~4,760 FLOPs at the average (T-s)/T ~ .4; by continuity ~640(T-s) ~ 2,550: 1.9x. V1's lines (T302's split: nonblind g23 3.9k, g12 3.5k, blind g23 2.3k per 720 triples) average 4.5 FPE, 2 of it the first cell. With B2 (2 -> .78) and continuity (2.5 -> 1.3) a line costs ~2.1 FPE: the walk part falls from 9.7k to ~4.6k. Phi/GL at G = 48 is ~2.4k FPE (T312), so C ~ 7.0k and HP2 F ~ 14.92 x 7.0/12.1 = **8.6 [6.2, 11.2]**: point pass, upper CI marginal. With G = 16 (the v2 integrand is smooth; the estimate moves ~1e-6 pp, which the DRI should rule a numerical change, not a variant), C ~ 5.4k and **F ~ 6.7 [4.8, 8.6]**: both criteria pass, with the HP2 errors already sealed.

*Evidence.* Continuity of a ReLU network across a gate flip is a theorem; T315's "uncapped CAP reproduces V1 to 0.0" shows the walk survives re-implementation; T313's B2 count is exact. The DRI ruling (cost counts only when executed) is met by implementing it.

*Decisive test.* Implement; assert max |est - V1| <= 1e-9 on all 80 DEV2; instrument FLOPs under Directive 8; T281 recounts HP2. Half a day.

*Stop rule.* Estimates differ by > 1e-9 pp (a bug), or instrumented C > 8.1k at G = 48: report the count and stop; nothing is lost.

### D2. Kinked-tangent surrogate (first-order event expansion of the line)

*Computation.* At the tie point of a line, run the first cell (value + tangent, B2-shared) and one adjoint per rival logit of the two smallest first-cell margins (near a 2~3 tie the dominant rival is p3; the adjoint of L_{p2} - L_{p3} is the same for the swap-pair class (p1,p3,p2), whose line passes through the same tie point, so it is shared). Each candidate event (s, i) with 0 < g_{s,i} = -z_{s,i}(0)/dz_{s,i}(0) < tmax contributes a kink: m(g) = m(0) + m'(0) g + sum_i lambda_{s,i} dz_{s,i} sigma_i (g - g_{s,i})_+, with sigma_i = +-1 for off->on / on->off. The correct set is solved on this piecewise-linear function (~160 kinks, sort, scan: ~1-2k FLOPs). Masses as in V1. Dropped: the second-order effect of early flips on the *locations* of later flips (the delta chains that D1 propagates). A ladder CAP-c + kinks (exact first c cells, then the surrogate from the c-th cell's gates) interpolates to the exact walk.

*Cost.* ~.4 (shared value) + .5 (tangent; on the g23 line the linearised lows move too, so it runs the full depth) + 1.3 (two adjoints, one shared) + .3 (kinks) ~ 2.5 FPE per line, independent of the cell count. For V1's three lines that equals D1's 2.1 (no gain); for the blind g12 line (14.7k FPE / 720 = 20 FPE each, dropped by V1 as "assumed correct") it is 8x; and every extra transverse point costs 2.5 FPE instead of 4.5-20. D2 does not beat D1 on V1; it buys what V1 could not afford (all four lines, two-sided pairing, a second transverse point) at V1's price.

*Evidence.* For: the kink structure is what T315 diagnosed ("the true margin stays correct after its gate flips"), and the kink weights are exactly what CAP-c discards; Hilton allows "seeing which weights fire on one input and extrapolating deductively". Against: flips at tiny g change later pre-activations' *slopes* by O(1) relative amounts (the shift scales with g, not with the tiny g_k), so the dropped cross terms are not small a priori, and T297's "more Newton steps make it worse" warns that first-order margin models misbehave here. Odds that the pure surrogate matches the walk within .3 pp RMS: ~40%; CAP-2 + kinks: ~60%.

*Decisive test (exact reference exists).* On the 10 T297 models compute the per-line correct sets with (i) `walk_line` (the reference, no MC needed), (ii) the surrogate, (iii) CAP-1/2 + kinks; report integration RMS vs the 1e7 MC of closure c (T302's protocol) and the per-line set mismatch. One to two CPU-hours.

*Stop rule.* Surrogate and CAP-2 + kinks both > .6 pp integration RMS (twice the walk's .28): the first-order event expansion is dead for the transition zone, scoped to lines and this closure, and T315 becomes dead-verified with the reason recorded. Go: <= .4 pp; rebuild V1 with all four lines two-sided and submit once to HP2.

### D3. Gaussian-smoothed closure verdict (the lows' first-order jitter): attack the floor, not the integral

*Computation.* The closure pins the 6-7 lows at mu(v4). Their true conditional law given the skeleton is iid truncated normal with model-free variance s^2(v4) = 1 - v mu - mu^2. To first order the margin at a skeleton point is m + J . (lows - mu), with J the margin's input-gradient at the low positions, so the exact-law verdict is Phi(m / (s ||J||)) instead of 1[m > 0]. Along a walked line m is affine per cell and J comes from D2's backward pass (lambda_s W_ih summed over low positions), so the correct mass per cell is the integral of Phi of an affine function against the gap density: a 1-D Gauss rule per piece, no network evaluations. The target changes: what the lows do becomes a smooth Gaussian functional of quantities the estimator already has, and the closure's error set becomes a band of the right expected width instead of a 0/1 set with a misplaced edge.

*Cost.* J once per line (~.9 FPE, declared constant along it) or per cell by a rank-1 update: ~1 FPE per line on top of D1/D2, plus ~.1 FPE of Gauss work per cell.

*Evidence.* Every closure overestimates accuracy by +.2-.3 pp on all 80 retrains and on LB3 (T298, T303, T304): the sign of an edge smeared where correct mass dominates. The one exact-law marginalisation tried, x9 alone (T307's B5), cut the floor 8%; the lows carry most of R0's cross-model spread (T298); T308's strongest cheap carrier is the spread of Phi(-m/8), a smoothed margin. The lows sit far from ties, where H-E15 says the partition is 37-44x coarser, so linearising in them is plausible. Against: 7 skewed summands are only roughly Gaussian, and gates driven by the lows are ignored. (T271's closed "FORM/SORM" was the input-space reliability index, a different object.)

*Decisive test (oracle, 15 CPU-min).* T298's paired protocol on 10 DEV2 models: per sampled x compare ok(x) with the 0/1 closure-c verdict and with Phi(m/(s||J||)) at the closure point (J by one backward pass). Baselines .471 (c) and .510 (R0). Go: >= 30% RMS cut (<= .33) with the mean offset halved. Stop: < 10%, as T307's rule.

*Why it matters.* Strict at 3k FPE allows .49 pp total. Floor .33 plus the walk's measured .28 integration (which D1 makes affordable) gives .43: strict becomes arithmetically possible in this lineage for the first time. Floor .47 makes it impossible at any integration accuracy, which is the current state.

## C. Answers to the brief's questions

(a) Event-centric: D2 is that decomposition; events (s, i) and their weights are shared across the swap pair (same tie point) and, since post-tie gate patterns are 92-100% set by the record history (FINDINGS §2), across classes with the same post-tie history (untested; at most ~2x on the adjoints). (b) The order of the first tiny cells is predictable in closed form only to first order (the ratios above); exact locations after interactions need the delta chains, which is D1's walk at half the price. (c) Along the wall: the second rung once D2 holds (two or three transverse points per region at 2.5 FPE each, or wall-direction tangents making every kink location affine in tau, so ell(tau) is a scalar root-solve per quadrature node with no network evaluation at the nodes). T306 B's failure (two points worse than one) is unexplained and must be diagnosed first: decompose V1's +.6 pp integration bias by source (tmax cut, +.17 from T302's scan; transverse point; product set S23 x S12; v4 linkage), each against the closure's MC. (d) D3 is the target change; the swap-pair identity "near-tie error = E_tau |ell(tau)| x rho(0)" is the other. (e) Plainly: I see no cheap deductive strict route outside D3 x D2. In the closure-and-walk paradigm the floor (>= .39 pp for any closure integrable under 10k FPE) excludes strict at <= 4k FPE unless D3 works, and T308/T314 say no function of the cheap anchor features reaches it. The scoped statement the evidence supports: *on retrains of the zoo configuration, every deductive estimator built either carries a closure floor >= .39 pp that costs >= 10k FPE to integrate, or extrapolates through the tie-wall transition zone at >= 2.7 pp of bias; the best (R1W V1) is within 1.5x of ARC's "roughly match" level on 100 sealed retrains (F 14.9 at 12.1k FPE, RMS .99, bias +.53) and fails strict at every budget; fitted ceilings show cheap per-class anchor features cannot carry strict.*

## D. Recommendation: continue, in this order

1. **D1 now** (one child, half a day, no new HP2 submission). Exact, no new modelling, and it plausibly moves the sealed R1W V1 score to a lenient pass on both criteria with the errors already known. Whatever else happens, this is the result to report to the owner and ARC.
2. **D3's oracle test and D2's line test in parallel** (two children, hours). Independent, each with a cheap exact or near-exact reference.
3. Then one pre-registered rebuild: V1 + continuity + whichever levers passed (all four lines two-sided if D2 holds; smoothed verdict if D3 holds; tmax 1.0, worth ~10% RMS in T302's scan and now affordable). One HP2 submission, pre-registered as a single variant (~36 DEV2 variants so far).
4. **Pivot rule.** If D2 and D3 both stop, the R0/R1W lineage is exhausted at F ~7-9 on HP2 (lenient, deductive, cancellation-free). Report that with the scoped statement above, mark T315 dead-verified, and stop funding cheap-regime variants. T270 stays the exact-route decision: its provisional lower bounds g = 8-11 at 58% coverage already sit above the "alive" line (<= 7), so I expect that route to close when the T316 fix lands; no compute beyond the approved budget.

## E. Odds (subjective)

Strict cheap (F <= 1 at <= 4k FPE) on sealed HP2 within 3 months: **7%** (was 5%; D3 is a new lever on the floor, but strict needs D3 and D2 both). Deductive, cancellation-free lenient pass on HP2, point and upper CI: **70%** (was 50%; D1 alone gets the point, G = 16 the CI). Strict at any cost via the exact route: **3%** (was 10%). Anyone, strict, within a year: 10%. ARC's own solution meets strict on retrains: 15%; meets "roughly match": 60%.
