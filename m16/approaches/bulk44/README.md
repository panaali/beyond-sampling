# T165: (4,4) bulk errors, the Hopf-fibre estimator, the T=4 H ladder, and T111

**Question.** What are the bulk error families of the 48-60%-accurate zoo (4,4) models? Can a deductive estimator that counts whole regions beat sampling there, and on (5,4) and (6,4) (owner directive)? T111: does the T87 recipe (estT) fail because T=4 or because the models are undertrained?
**Status:** done. **Method (deductive, no random inputs).** The model has no bias, so the prediction and the label do not change under x -> c x. Accuracy is therefore a uniform measure on S^3. S^3 is the union of Hopf great circles x(phi) = a cos phi + b sin phi, and these circles are spread uniformly over (u, d) in [0,1] x [0,2pi). On each circle every pre-activation, logit difference and input difference is A cos + B sin on each linear piece, so one atan2 gives its zeros. The circle is cut **exactly** into pieces where gates, prediction and label are all constant (the bulk families are whole arcs, and the boundary wedges are included). The correct arc is summed. A fixed midpoint grid of nu x 2nu circles integrates over (u, d). Every coefficient comes from the weights. Files: `hopf.py` (estimator), `check_fibre.py` (exact circle vs a 2e6-point grid: max error 1.4e-6 over 40 circles), `ev.py` (panel), `report.py` -> `report.json`, `fam.py`/`cham.py`/`sign.py`/`cells.py` (MC diagnostics -> `fam_4_4.json`, `cham_4_4.json`, `sign_4_4.json`, `cells_4_4.json`), `pieces.py`, `conv.py`.

## Mechanism of the (4,4) bulk errors (MC 1e6, `fam.py`, `cham.py`, `cells.py`) [measured]
- **s0 (0.477):** whole ordering chambers fail. If x1 is the max and x0 the 2nd, it answers 1. If x2 is the max and the 2nd is x0 or x3, it answers 2. In both cases it reports the argmax. It is wrong 75% / 63% of the time when x0 / x3 is the 2nd-largest. 69% of its errors are the argmax.
- **s1, s2, s4 (0.60):** these are **the same solution up to a neuron permutation** (max weight difference <= 0.07). No chamber is more than 80% wrong. 58% of inputs end in the all-on cell, and that cell is 40% wrong: in that one linear regime the readout cannot separate the 2nd input from the 1st or 3rd (errors are 48% argmax, 42% third). The worst chambers (max..min order) are 1023, 3021, 2310 and 3210, at 73-80% wrong.
- **s3 (0.498):** 6% of inputs end in the all-off cell, and 96% of those are wrong. All logits are 0 there, so argmax ties to class 0. Chambers 2310->0, 3012->1, 3021->3 and 3210->1 are >80% wrong.
- On every seed the errors are spread over hundreds of gate sequences: the top sequence holds <= 8% of errors (`sign_4_4.json`). That is why freezing intermediate-step gates (estT) biases the result: the error is not concentrated in a few boundary wedges.

## Results (`ev.py 44|54|64|43k20; report.py 44 54 64 43k20`)
For each seed the score pools the clean model with 4 ARC noisy copies (abs sigma 0.05, noise seed 50). The ratio is the MSE over the sampling MSE p(1-p)/(ops/ops_per_sample) at equal ops. Truth is MC 1e7, with **no** subtraction of the truth noise, so the ratios below are conservative upper bounds. The frozen recipe is nu=64 (8192 circles) for every T=4 rung.

| rung (zoo acc) | Hopf nu=64 ratio, seeds 0-4 | estT recipe ratio | Hopf ops | pieces/circle (vs estT exact cells) |
|---|---|---|---|---|
| (4,4) 0.48-0.60 | 0.10 / 0.020 / 0.055 / 0.059 / 0.025 | 161 / 55 / 21 / 140 / 18 (fails) | 63M | 68-72 (estT exact 1897-2595) |
| (5,4) 0.64-0.75 | 0.049 / 0.41 / 0.12 / 0.066 / 0.30 | not run | 94M | 76-85 |
| (6,4) 0.60-0.86 | 0.29 / 0.11 / 0.55 / 0.040 / 0.49 | not run | 131M | 88-95 |
| (4,3) k=20 0.57-0.80 | nu=1024 meridians: <= 0.0033 | **0.0034 / 0.0008 / 3e-5 / 0.069 / 6e-5 (passes)** | 2.0M / 0.09M | - |

Against the high-resolution Hopf reference (nu=128) the (4,4) ratios are 0.006-0.038. Check on (4,4) s0: Hopf nu=256 gives 0.477036 and MC 2e8 gives 0.477014 +- 3.5e-5. The estT exact value from run87, 0.47691, is 3.0 SE low, which agrees with the T131 leaf-mass bias at T=4.

**Verdict.** All three T=4 rungs pass: every seed beats sampling at equal ops. The worst ratio grows with H: 0.10 -> 0.41 -> 0.55, so the margin is shrinking, and (7,4) or (8,4) may fail at nu=64. **T111:** on undertrained (4,3) models of similar accuracy (0.57-0.80), the same estT recipe that fails at (4,4) passes on all seeds. So the (4,4) failure comes from T=4 (frozen step-2/3 gates over bulk errors spread across many sequences), not from undertraining alone.

## Claims
- [measured] The Hopf-fibre estimator beats sampling on all 5 seeds plus 4 noisy copies at (4,4), (5,4) and (6,4) (table, MC-1e7 truth, conservative).
- [measured] estT recipe: (4,4) ratio 18-161, while (4,3) k=20 is <= 0.069. Seeds 1, 2 and 4 of (4,4) are one solution up to neuron permutation.
- [inferred] The circle integrand F(u,d) is continuous, so the grid error falls about as N^-1. The observed convergence is irregular at the 1e-4 level (nu 64/128/256: .47710/.47717/.47704).
- [inferred] T111: the failure is about T (gate freezing at steps 2..T-1), not undertraining. The accuracies are matched only roughly (0.57-0.80 vs 0.48-0.60).

## Compute accounting
Ops follow the arc_m43_repro `Ops` convention (multiply-add, atan2, compare and cos each count 1): about 7.6k ops per circle at (4,4), 11k at (5,4) and 16k at (6,4). Sampling is charged T·H·(H+1)+T·H ops per sample, as in run87. The MC truth (1e7 samples per model, 75 models) is not charged.

## Alternatives not ruled out / failure modes
The ratios carry MC-1e7 truth noise. The Hopf reference disagrees with MC by up to 3.6 SE at (6,4), so its own error is about 3e-4 there. The pooled per-seed MSE uses only 5 models. The estT recipe was not run at H=5 or 6. The method needs T=4 for Hopf or T=3 for meridians (for odd or larger T, use another circle fibration, e.g. pairs of complex coordinates when T is even). The cost per circle grows with H and T (the pieces grow), and at larger T the base grid is (T-2)-dimensional, so the advantage over sampling can vanish.

## Exhaustion checklist
| variant | status |
|---|---|
| Hopf grid nu = 8/16/32/64 (and 128/256 as reference) | done (`ev_*.json`) |
| estT recipe at (4,4), reused from run87 for the same models and truth | done |
| estT recipe at (5,4), (6,4) | not run (cost; Hopf is the frozen recipe for this ladder) |
| fixed T131 exact reference | not available; MC 1e7 (2e8 on s0) used instead |
| (4,3) k=20 recipe and Hopf meridians | done |
| better-trained (4,4) (T86) | not done (T111 answered via (4,3) instead) |
| region-family closed-form cone masses | not built: the circle method counts whole arcs exactly, so cone masses are unnecessary |
| higher-order / Richardson quadrature in (u,d) | not tried (backlog idea) |
