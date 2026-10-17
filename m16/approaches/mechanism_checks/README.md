# T8: phase clocks and orthogonal mode checks

## Question
Does H-M1's phase clock survive other common gate patterns? Does an orthogonal and causal check support H-M3's attribution of n3/n13 to faster rotating modes?

## Method
Planned before experiments: select the five commonest gate patterns at steps 6–10 using a separate Gaussian pilot. On a fresh evaluation set, measure final-state phase versus age of the positive running maximum separately for each pattern. Compare observed phase slope with its own slowest rotating eigenvalue, report circular concentration and a shuffled-age control. Retain minimum support and every selected pattern, even if a pattern lacks enough ages.

For each pattern, orthonormalize the real invariant span of the next two complex pairs after the slowest pair. Report n3/n13's projection leverage and its rank among all neuron pairs. Unlike the original nonorthogonal variance shares, these coordinate leverages are bounded in [0,1]; they describe geometry, not variance attribution. Compare orthogonal fast-subspace removal with equal-rank random subspaces at the final state, using a pilot conditional mean, and compare n3/n13 mean-ablation with other neuron pairs. All ablations are restricted to observed states in the matching gate pattern and cannot establish an in-distribution causal circuit.

### Alternatives written before experiments
1. The apparent phase clock is specific to the most common pattern, or gate conditioning confounds age. Cheapest discriminator: independent pilot selection, per-pattern evaluation with age support, predicted versus observed slope, and within-pattern shuffled ages. This does not rule out conditioning on the full gate history.
2. Large n3/n13 mode shares are a nonorthogonal-coordinate artifact. Cheapest discriminator: bounded Euclidean projection leverage onto a real fast invariant subspace, ranked against all coordinate pairs, plus overlap with the slow subspace.
3. Ablation damage is generic loss of high-energy/readout-important directions, not fast position coding. Cheapest discriminator: equal-dimensional random orthogonal and neuron-pair controls, reporting removed hidden-state energy alongside accuracy loss. Energy/readout matching and on-manifold patching remain separate stronger follow-ups.

### Recorded definition correction after the first run
The literal slowest rotating eigenpair is not always the dominant oscillator used by `modes.py`: e.g. the most common gate has a second, more damped pair at period 8.37 alongside the 8.42 clock. The first implementation also selected that pair as a "higher" mode by angle order. Its entire output is retained as `results_slowest_exploratory.json`; it is not the headline H-M3 test. Before further runs, the operational definition is changed to the complex pair with largest modulus, matching `modes.py`'s magnitude ordering; the fast span comprises the two largest-modulus pairs with greater phase angle. Both definitions are reported, so the sensitivity is visible. A second fresh seed will check stability under the corrected definition. This is a diagnostic correction, not a preregistered confirmation.

## Status
Done: bounded diagnostics on the original M_16,10, two fresh evaluation seeds. H-M1 extends as an approximate phase/age relationship in some common patterns, not a universal constant-rate clock. H-M3 gains orthogonal geometric evidence, but causal specificity is **not demonstrated**. No claim is self-confirmed. Parent coordinates registry and task-board updates.

## Results

Each run selects gate patterns on 10,000 pilot sequences and evaluates 40,000 new sequences. Seeds 82026 and 82027 select the same five patterns. For seed 82026 they cover 39.138% of pilot states at steps 6–10 and 57.373% of final states. Overall accuracies are 95.245% and 95.218%; these sampling estimates are diagnostics, not replacement ground truths.

### H-M1: a qualified clock survives, but not the literal slowest-mode rule

Angles below are degrees per **age since the maximum**, so the sign reverses when expressed per maximum position. Phase slopes are unweighted least squares on unwrapped circular means of age bins with at least 50 examples; three populated ages are required. The eigenvalue prediction is the angle of the largest-modulus complex eigenvalue. No slope tolerance was preregistered.

| Gate bitmask | Pilot share | Final positive-max n | Supported ages | Predicted slope | Observed, seed 82026 / 82027 | Dephased concentration / shuffled median, seed 82026 |
|---|---:|---:|---|---:|---:|---:|
| 64975 (original STEADY) | 16.824% | 11,468 | 3–9 | 42.75 | 41.47 / 41.71 | .952 / .054 |
| 63951 | 6.760% | 3,447 | 6–8 | 46.57 | 38.37 / 38.86 | .963 / .798 |
| 65519 | 5.988% | 2,591 | 1 | 53.61 | insufficient support | .920 / .919 |
| 62927 | 5.066% | 2,456 | 5 | 47.56 | insufficient support | .907 / .885 |
| 65007 | 4.500% | 2,967 | 1, 3, 4, 7–9 | 42.84 | 47.03 / 47.02 | .825 / .129 |

The original pattern's binwise concentrations are .890–.982 on supported bins. The mismatch in 63951 is repeatable (~8 degrees), and 65007 differs by ~4 degrees: a common phase/age association does not imply that the final gate matrix's eigenvalue determines the full trajectory. The shuffled-age test has minimum attainable p=.005 (199 permutations); 64975, 63951, 62927 and 65007 attain that value in both seeds. This is a within-pattern association control, not proof of causal timing. In particular, the 62927 statistic is driven by rare ages outside its single well-populated bin and does not establish a slope. Pattern 65519 has p=.240/.775.

The retained literal-slowest-mode exploratory run is negative on 65007: observed 2.90 degrees versus 33.58 predicted, concentration .169 versus shuffled .212. The dominant oscillator there instead has period 8.40; the slower 10.72-period pair has smaller modulus. A clock claim must specify which mode it means. In the original gate, the exploratory "fast" selection accidentally included the more damped 8.37-period pair; that output is retained and superseded by the explicit mode rule above.

### H-M3: geometric enrichment, without demonstrated causal specificity

We QR/SVD-orthogonalize the real span of the two strongest complex pairs faster than the dominant oscillator. Projection leverage is the fraction of a coordinate unit vector lying in that four-dimensional span. It is **not** a fraction of neural activation variance. Pair ranks compare mean leverage against every active neuron pair (rank 1 highest).

| Gate | Fast periods | Mean n3/n13 leverage | Pair rank | Slow/fast squared overlap (0–2) | Fast ablation drop / random median (pp) | n3/n13 ablation drop (pp) |
|---|---|---:|---:|---:|---:|---:|
| 64975 | 4.32, 2.89 | .751 | 2 / 78 | .093 | 60.32 / 72.40 | 30.28 |
| 63951 | 4.70, 3.06 | .700 | 6 / 66 | .905 | 57.12 / 75.20 | 26.40 |
| 65519 | 2.06, 3.99 | .642 | 5 / 105 | .447 | 44.11 / 67.91 | 29.37 |
| 62927 | 2.88, 3.55 | .840 | 1 / 66 | .755 | 57.41 / 71.50 | 31.27 |
| 65007 | 3.91, 3.08 | .682 | 4 / 91 | .363 | 40.51 / 62.74 | 19.72 |

Ablations replace the selected final-state component by its pilot, gate-conditional mean. Random controls use 32 independent four-dimensional subspaces inside the active coordinates. On seed 82026, all 32 random controls cause more accuracy damage than the fast subspace in every pattern. On seed 82027, 32/32 do so in four patterns and 31/32 in 62927. But fast removal deletes only 11.5–16.8% of centered hidden-state energy; random median removals delete 22.9–32.3% on seed 82026. Therefore this difference cannot establish either positive or negative causal specificity. Neither controls nor target are matched on readout gain or on-manifold plausibility.

The n3/n13 pair is near the least damaging end of same-size active-neuron pair ablations (ranks by damage 77/78, 64/66, 98/105, 64/66, 90/91), despite being enriched geometrically. These interventions do show functional relevance, but do not show that the pair specifically resolves high spatial frequencies. Other pairs include essential comparator neurons and remove different energies. Orthogonal projections may generate negative states; JSON records this fraction. Even nonnegative mean-ablated states can be off manifold.

### Reproduction and validation

From the repository root, with numpy, torch and alg_zoo installed (this session used `/workspace/clean-neurons/.venv/bin/python`; no new package installs):

```bash
curl -fL --max-time 45 https://storage.googleapis.com/arc-ml-public/alg/one_layer_16_hidden_94_acc_2nd_argmax.pth -o m16/approaches/mechanism_checks/checkpoint.pth
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python m16/approaches/mechanism_checks/check.py --weights m16/approaches/mechanism_checks/checkpoint.pth --mode-selection slowest --output m16/approaches/mechanism_checks/results_slowest_exploratory.json
python m16/approaches/mechanism_checks/check.py --weights m16/approaches/mechanism_checks/checkpoint.pth
python m16/approaches/mechanism_checks/check.py --weights m16/approaches/mechanism_checks/checkpoint.pth --seed 82027 --output m16/approaches/mechanism_checks/results_seed82027.json
python m16/approaches/mechanism_checks/validate.py
```

The public checkpoint is 3,385 bytes, downloaded unchanged and kept uncommitted. SHA256: `498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb`. Converted float64 arrays have SHA256 `f8d3dc132faf534307ece33a56b9b3d8238975e4d7e59936ab2c5304c7fd2622`. No external ARC prose was read. The initial default-sandbox model load could not access sockets; the bounded checkpoint download succeeded under authorized escalation.

`validate.py` independently compares the NumPy recurrence against AlgZoo's PyTorch `DistRNN` on 128 fresh inputs: max absolute logit discrepancy **2.20e-13**. It also recomputes projection leverages with QR instead of SVD: all ten pattern/seed rows agree within 1e-10, leverages remain in [0,1], and each sum is 4. `validation.json` records the checks. These checks verify implementation, not independent scientific confirmation.

## Compute accounting

This is a **sampling-based mechanism diagnostic**, not an ARC estimator. No leaderboard claim is made. All three experiment runs (including the retained exploratory run) total **150,000 full model sequences**, 1,500,000 recurrent steps, **7,920,413 readout rows**, 1,525,344 Gaussian variates, 15 eigendecompositions, 510 SVDs, and 13,660,554 rows in phase-shuffle operations. Reusing a seed still counts as repeated computation. Validation adds 256 sequence equivalents (128 each through NumPy and PyTorch), 1,280 Gaussian variates, 10 eigendecompositions and 10 QR factorizations.

The recorded arithmetic subtotal is **4,694,968,128 FLOPs** including validation recurrence/readout: recurrence uses 560 FLOPs/step, readout 320 FLOPs/row, projection 4*N*16*rank, Gaussian generation convention 20 FLOPs/variate. Validation input-generation cost adds 25,600 FLOPs. This subtotal excludes sorting, comparisons, reductions, complex phase/shuffle operations and matrix decomposition/inversion FLOPs; their counts and measured runtime are reported instead. It must **not** be used as a full leaderboard cost.

Measured experiment-kernel CPU times are .632 + .638 + .619 seconds; validation adds .017 seconds, total **1.905 CPU-seconds excluding imports**. Whole Python invocations took about seven seconds wall time in total. Dependency probes, one eigenvalue diagnostic, JSON inspection, imports and checkpoint transfer were additional small overhead not instrumented with a full process CPU counter. No training, LP solves, or heavy jobs were run; all numerical work used one thread, far below the ten CPU-minute limit. JSON preserves exact per-run counts and timing.

## Claims

- **[measured]** The original common gate exhibits a strong phase/age relationship with a slope close to its eigenvalue prediction on both fresh evaluation seeds; two additional patterns have multi-age associations with repeatable slope departures. Two selected patterns cannot establish a slope with the available support.
- **[measured]** n3/n13 is enriched in an orthogonal fast invariant span across all five patterns: mean coordinate leverage .642–.840, pair ranks 1–6. This repairs the mathematical attribution metric, not the broader harmonic interpretation.
- **[measured]** Fast-space and n3/n13 mean-ablations damage predictions, but equal-rank random subspaces and most other neuron pairs damage them more under unmatched-energy interventions.
- **[inferred]** H-M1 should remain qualified: the dominant-mode phase is useful positional information in several common configurations, not evidence of a universal gate-independent clock.
- **[inferred]** H-M3's geometric component gains support; its causal claim that n3/n13 supplies higher-harmonic positional resolution remains unproven. These interventions neither establish nor refute causal specificity.
- **[speculative]** On-manifold counterfactual swaps and energy/readout-matched controls could separate high-frequency position information from generic output sensitivity.

## Alternatives not ruled out

Input forcing and the complete gate history can explain the deviation from a final gate's phase increment. Gate conditioning can itself select ages. Euclidean orthogonality depends on neuron scaling; only the trained parameterization was tested. Fast and slow invariant subspaces need not be orthogonal (overlap reaches .905 out of 2). The faster pairs are not universally integer harmonics. Random subspaces are not matched for energy or readout gain. Strong off-manifold damage is not a selective mechanistic test. Two sampling seeds from one checkpoint are not independent training-seed replication.

## Failure modes

Small gate/age strata; unsampled gate patterns; choice of mode selector; unwrapped phase slopes across gaps; nonnormal recurrence; gate changes; off-manifold interventions; phase correlations without recurrent causality. Statistical permutation p-values have finite resolution and do not correct any causal confound. There is no confidence interval for the phase slope, and the two-seed range is a stability check rather than a confidence interval.

## Suggested registry / ledger / follow-up rows (parent integration)

Registry: **T8 / Q3, done, verification S** — original checkpoint, two seeds: phase clock remains strong but gate-dependent; orthogonal fast-span n3/n13 leverage .642–.840 (top 1–6 pairs), causal specificity not demonstrated by energy-unmatched mean ablations. Folder: `m16/approaches/mechanism_checks/`.

H-M1: keep **supported, qualified**, replace "not checked per gate pattern" with five-pattern evidence, three patterns with adequate age support, deviations up to ~8 degrees/step, and two patterns lacking support. H-M3: retain **weak/withdrawn as a causal harmonic claim** while recording orthogonal geometric enrichment; do not promote to confirmed.

Suggested new Ready light follow-up (allocate next unused T-ID): **On-manifold fast-position causal test** — pair inputs with controlled swaps of second/third-largest positions, patch n3/n13 and fast-mode components, compare energy/readout-matched controls and score target-specific positional mediation. Done when selective resolution is demonstrated or its absence bounded, with manifold validity documented. This addresses the remaining causal limitation, not generic ablation damage.

Suggested new Ready light follow-up (allocate next unused T-ID): **Gate-history phase transport** — test the exact eigen-coordinate recurrence including input forcing on runs of unchanged gates, and compare full gate-history transport to a last-gate-only phase clock. Done when the ~4–8 degree slope discrepancies are explained or persist under matched histories. Start with this folder's five patterns; no need to repeat the global search.
