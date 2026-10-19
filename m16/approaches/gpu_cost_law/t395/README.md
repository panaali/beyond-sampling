# t395: 1e10-input census of error-carrying and near-tie gate patterns of M_16,10 (T395, extends T348)

**Question.** How many distinct gate patterns (activation regions) carry the error mass and the near-tie mass of
M_16,10 at prefix lengths k = 7..10, how much of that mass do the observed patterns cover, and how much is unseen
(Good-Turing)? T348 (1e9 inputs) could not resolve error patterns beyond k ~ 7-8 (coverage .96 at k = 8, .86 at k = 10);
every "dead" verdict on exact routes for rare events leans on this growth.

**Status.** DONE 2026-10-19 (A100 run 1625 s; 2.73 of 10 units; runtimes stopped). Pre-registered 2026-10-19 (the
pre-registration was written and pushed before any T395 run). Owner: Claude session_016BGedpds19gq95yjWmDMn5
(T192 child), DRI session_01CD3g9Cvc3qk3pM11eyXXzj. Cap: 10 Colab units (on top of T352). No ARC contact, no Fable.

## Pre-registration (fixed before any result)
- **Model and inputs:** M_16,10 (`m16/arc_eval.py` weights), x ~ N(0, I_10), 1e10 inputs (CUDA Philox, one seed,
  regenerated identically in every pass), float64 forward, gate word w_t = 16 bits of h_t > 0 (as T348).
- **Sets:** error inputs (argmax(Wo h_10) != position of the 2nd-largest x) and near-tie inputs (min(g12, g23) < .1);
  plus all inputs for validation.
- **Patterns:** prefix (w_1..w_k) for k = 7..10 on error and near-tie sets; prefix k = 4..6 on all inputs (validation);
  64-bit order-dependent hash as T348 (exact 64-bit key for k <= 4); collision check with a second independent mixer at
  k = 10 on the error set.
- **Counting:** GPU sort + unique in hash partitions sized to the GPU memory; per (set, k) the full count-of-counts is
  SAVED (not only f1), so coverage-adjusted estimators can be computed afterwards.
- **Reported per (set, k):** n, distinct D, singletons f1, doubletons f2, Good-Turing coverage C = 1 - f1/n (unseen mass
  1 - C), Chao1 richness D + f1^2 / (2 f2), plug-in exp(H) and the Chao-Shen coverage-adjusted entropy exp(H_CS),
  N(delta) = fewest patterns covering 1 - delta of the set's sample mass for delta in {1e-1, ..., 1e-6} where
  delta * n >= 10 (flagged unresolved where 1 - C > delta / 3), and per-step growth factors of D, exp(H), exp(H_CS),
  N(delta).
- **Validation (pass/fail recorded):** k = 4, 5 all-input exp(H) and N(delta) vs the exact enumeration
  (`../results/prefix/k{4,5}_regionmass.npz`): |relative difference| <= 1% where resolved; k = 6 distinct D <= exact
  region count 19,740,596 (T270, a lower bound itself) and N(delta) vs T270's region masses (provisional: T270's masses
  are lower bounds for defective cells).
- **Decision reading (as framed by T346):** error-pattern growth per step (exp(H_CS) and resolved N(delta)) <= ~3x ->
  exact sub-cone enumeration revives for rare events; ~25-44x -> exact routes dead; in between -> report as such.
- **Compute plan:** A100 (float64), partitions P chosen so a pass holds <= ~25 GB of keys; estimate ~1 h of A100
  (~5.3 units) + setup; stop at 10 units and report the k reached.

## Method
`census10.py N P` (CUDA, float64 forward, Philox seed 3950000, batches of 2^22 regenerated identically in each of P hash
partition passes; per-batch `th.unique` with outputs cloned; in-pass compaction of each job's buffer above `COMPACT`
keys; per-partition checkpoint and resume). Compaction and interrupt + resume were checked on CPU to give results
identical to a one-shot run. `validate.py` checks the pre-registered validation. Run: `q_t395.sh` on an A100-40GB,
`COMPACT=5e7 census10.py 1e10 6`, 6 passes of ~271 s. Output: `census10.json` (full count-of-counts per (set, k)),
`validate.json`.

## Results ([measured], unverified)
n = 1e10 inputs: 468,826,199 errors (4.69%), 3,440,247,984 near-ties (34.4%).

| set, k | distinct D | coverage C | Chao1 | exp(H) | exp(H_CS) | N(0.1) | N(0.01) |
|---|---|---|---|---|---|---|---|
| err 7 | 3,612,737 | .9969 | 5.79e6 | 34,373 | 34,903 | 109,655 | 1,106,479 |
| err 8 | 10,661,344 | .9891 | 1.96e7 | 131,021 | 135,852 | 551,125 | 5,973,083 (u) |
| err 9 | 24,480,981 | .9709 | 5.21e7 | 416,088 | 440,524 | 2,431,458 | 19,792,720 (u) |
| err 10 | 43,365,448 | .9431 | 1.05e8 | 1,129,783 | 1,192,548 | 8,122,886 (u) | 38,677,187 (u) |
| near 7 | 6,839,935 | .9993 | 1.00e7 | 37,253 | 37,420 | 108,958 | 1,036,987 |
| near 8 | 24,785,858 | .9970 | 4.10e7 | 145,841 | 147,989 | 568,949 | 6,839,978 |
| near 9 | 68,763,101 | .9902 | 1.29e8 | 499,666 | 516,138 | 2,643,243 | 34,635,424 (u) |
| near 10 | 149,889,235 | .9756 | 3.23e8 | 1,518,767 | 1,593,788 | 10,764,919 | 115,486,756 (u) |

(u) = unresolved by the pre-registered rule (1 - C > delta / 3); an unresolved N(delta) is a lower bound.
Collision check: the second mixer at err k = 10 gives identical D, f1 and every statistic.

**Growth per step** (k-1 -> k, for k = 8, 9, 10). Error set: exp(H_CS) x3.89, x3.24, x2.71; D x2.95, x2.30, x1.77
(D is sample-limited); resolved N(0.1) x5.0, x4.4 (k = 8, 9). Near-tie set: exp(H_CS) x3.95, x3.49, x3.09; resolved
N(0.1) x5.2, x4.6, x4.1.

**Validation** (`python validate.py census10.json`):
- k = 4 PASS. exp(H) 301.7 vs exact 301.7. N(delta) within 0.6% for every delta (all resolved).
- k = 5 FAIL on one entry. exp(H) 1545.6 vs exact 1545.6, and N(delta) within 0.3% for delta >= 1e-4. But N(1e-5) =
  149,034 vs exact 152,956 (-2.6%) although the rule calls it resolved (1 - C = 3.1e-6 <= 3.3e-6). The pre-registered
  resolution rule is too loose for N(delta): the singleton mass is not the only thing missing, because tail patterns
  seen 2-5 times are under-counted too. The failure is recorded as is. Resolved entries with delta >= 1e-4 are reliable.
- k = 6 PASS: D = 1,651,248 <= 19,740,596 (T270 region count; most of T270's regions carry tiny mass); exp(H) 7,156.

**Decision reading (pre-registered framing):** error-pattern growth per step is 2.7-3.9x in exp(H_CS) and 4.4-5.0x in
resolved N(0.1). That is far below the ~25-44x "dead" band, and close to (slightly above) the ~3x "revives" line. The
exp(H_CS) factor falls with k (3.89 -> 3.24 -> 2.71). By the pre-registered rule this is "in between", nearer the revival
end. The count needed for small delta is not resolved at k >= 9: N(0.01) at k = 10 is >= 3.9e7 error patterns.

## Compute accounting
Colab: T4 dry run at 1e8 (~0.1 units incl. setup), A100 run 1625 s plus setup: 2.73 units in total (balance 150.08 ->
147.35), of the 10-unit cap. Model forwards: 6 passes x 1e10 = 6e10 float64 forwards (each pass regenerates the inputs).
CPU: compaction, resume and validation tests at 3e5-1e7 inputs; the analysis.

## Claims
- [measured] At 1e10 inputs, error-carrying gate patterns number >= 4.3e7 distinct at k = 10 (coverage .943); their
  effective count exp(H_CS) grows ~2.7-3.9x per step from k = 7 to 10, and N(0.1) ~4.4-5x per resolved step.
- [measured] The census reproduces the exact k = 4, 5 entropies to 4 digits; N(delta) agrees within 0.3% where
  delta >= 1e-4.
- [inferred] Error mass is not spread evenly over the ~5^k regions. Its effective pattern count grows at about 3x per
  step, the threshold the pre-registration used for exact sub-cone enumeration to stay viable for rare events, while
  N(0.01) (>= 3.9e7 at k = 10) is already large.

## Alternatives not ruled out
- Growth may change beyond what 1e10 resolves: the unseen mass at k = 10 is 5.7% (error) and 2.4% (near-tie), and
  N(delta) for delta <= 0.01 is unresolved at k >= 9 (lower bounds only).
- exp(H) measures the typical spread; the cost of an exact method depends on N(delta) at the target delta, which
  grows faster (4-5x) than exp(H).
- The "error" set is defined by the network's argmax vs the true 2nd-largest position; rare-event targets (far-gap
  picks, T340/T398) are a tiny subset whose pattern growth may differ.

## Failure modes
- The resolution rule 1 - C <= delta / 3 is too loose for N(delta) (k = 5 validation entry at delta = 1e-5).
- Hash collisions at 64 bits: the second mixer agreed exactly; with 1.5e8 keys the expected collisions are ~6e-4.
