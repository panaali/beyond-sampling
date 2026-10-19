# walk_rules (T442): is the T = 10 strict-bar result robust to the quadrature rule, and does a better rule help?

↑ [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) (H-E17, T429, T431) · [novelty review s4 rank 7](../../../lit/novelty_review.md) ·
base: T429 `m16/approaches/walk_budget/` @ 79ecae63 (branch claude/t429-walk-budget)

## Question
H-E17 (the frozen T10 walk quadrature beats plain sampling at equal compute on M_16,10 retrains from 1e6 FPE; sealed
HP5 F .54 at 4e6) rests on ONE scrambled-Sobol rule seed (31415). (1) Was that seed typical? (2) Does another rule
family (rank-1 lattice, with or without periodisation; Halton) do reliably better? (3) What is each rule's scaling
exponent? Not in scope: a weight-derived error bound (T449).

## Pre-registration (written 2026-10-19, before any scoring of a new rule)
- **Code:** T429's runner and frozen T10 recipe, verbatim: `t429_score.py` = walk_budget/score.py (sha256[:12]
  afcb43325500, asserted in `run.py`), which itself asserts the seven frozen-file hashes (estimator.py fb7902e12240,
  core.py 48dbba49ab13, walk.py a38258f586b4, hom.py a91f019fca21, walk_nb.py bbc6cc992c7c, ...) and the DEV3 data
  hashes (weights 62c952037e8c, truths abcb7d6d264a). T393 compiled walk (repro mode), T429's lazy Sobol rule.
- **The only change is the line set** (`run.py`): `C.sobol(T, nmax, RULE_SEED + 101 j)` for stratum j is replaced by
  another rule factory with the same call and prefix-slice interface; `est(..., rule_seed=SEED)` sets RULE_SEED.
  Strata, face maps, pilot (25 %, power-of-2 prefixes), Neyman allocation, walk and cost are untouched.
- **Rules (configs.py), 27 new configurations + the frozen one:**
  - (a) `sobol`: 10 new scrambled-Sobol rule seeds (`np.random.default_rng(442).integers(1, 2**30, 10)`); the frozen
    seed 31415 is reused from T429's saved rows (gate: bitwise reproduction).
  - (b) rank-1 lattices: the Kuo / Cools-Kuo-Nuyens embedded generating vector `lattice-33002-1024-1048576.9125`
    (first 10 components, file `kuo_lattice-33002-1024-1048576_first32.txt`, taken from the qmcpy 2.4 wheel because the
    UNSW site was unreachable), points in radical-inverse (extensible) order, so every power-of-2 prefix is a full
    lattice; <= 2^20 points per stratum (max used at 4e6 FPE ~2.7e5). Three rules:
    `lat` (random shift, 6 shift seeds), `latb` (same shifts + baker's / tent periodising transform; paired with
    `lat`), `lat0` (fixed: no shift, no transform; one rule).
  - (c) `halton`: scipy scrambled Halton, 4 seeds.
- **Panel:** DEV3 (100 open retrains, T342), T429's 1e8 truths. **HP5 is not touched** (spent); any later sealed run
  needs a fresh panel and a DRI pre-registration.
- **Budgets:** 1e6 and 4e6 FPE (primary) plus 256k (for the exponent only). One deterministic run per model, config, budget.
- **Metrics (`analyse.py`, fixed now):** F exactly as T429 (truth variance subtracted), 95 % cluster-bootstrap CI over
  models (2000 resamples, rng 0, the same resamples for every config, so comparisons are paired). Both bars per config:
  strict cleared iff CI upper < 1, lenient iff CI upper < 10 (PROTOCOLS Directive 8).
  Per family: mean F over its seeds (= expected F of a randomly drawn rule of that family) with bootstrap CI, the
  across-seed sd and range, and the count of seeds clearing the strict bar.
  Exponent per config: slope of log F on log mean cost over 256k / 1e6 / 4e6, bootstrap CI; family mean and sd.
- **Reading rules:**
  - **Frozen seed typical** iff its F ranks neither best nor worst among the 11 Sobol seeds at that budget (also
    reported: z vs the 10 new seeds). Atypical at both primary budgets → H-E17's single-seed evidence is flagged as
    seed-selected / lucky (or unlucky) to the DRI.
  - **Robustness of H-E17 to the rule seed:** the Sobol family mean F's CI upper < 1 at a budget ⇒ the strict-bar
    result holds for a random Sobol rule there (not only for 31415).
  - **Family better / worse** than Sobol: paired-bootstrap ratio of family-mean F to Sobol-family-mean F; BETTER iff
    the ratio CI upper < 1 at both 1e6 and 4e6, WORSE iff the CI lower > 1 at both, else NOT DISTINGUISHED.
    (lat0 is a single fixed rule, so its ratio carries no rule-spread uncertainty: read with that caveat.)
  - Novelty-review stop rule: a lattice family with CI upper on F < .5 at 4e6 is a "gain"; otherwise "no gain".
- **Predictions:** frozen seed typical; Sobol seed sd at 4e6 ~ .1-.15 in F (comparable to the model-bootstrap
  CI half-width); no family BETTER (integrand is discontinuous in the configuration coordinate and at the walk's
  error-set boundary, so lattice + baker's gains for smooth periodic integrands should not appear); exponents
  -.2 to -.35 for every family.
- **Threats noted in advance:** (1) one deterministic run per model per config: model bootstrap only; seed spread is
  the rule uncertainty; (2) the Neyman allocation is computed from each rule's own pilot, so a rule changes the
  allocation too (part of "the rule", as in T429); (3) truth noise is ~8 % of raw squared error at 4e6; (4) 10
  Sobol seeds give a coarse distribution (rank test only resolves extremes).

## Method
- `run.py FAMILY SEED BUDGET OUT.jsonl`: one `est` call per DEV3 model (resumable). `drive.sh W K`: the queue
  (256k and 1e6 for all configs first, then 4e6). `gate.py` → `gate.log` (passed before scoring: sobol 31415 bitwise
  equals T429's saved rows; lattice prefix consistency and full-lattice 2^m prefixes; Halton prefix consistency).
- `analyse.py` → `analysis.json`, `analysis.log`.

## Status
Pre-registered; scoring not started.
