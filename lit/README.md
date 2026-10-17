# lit/: literature scans

↑ [Project README](../README.md) · [STATUS](../STATUS.md) · [INDEX](../INDEX.md) · [Approaches](../m16/approaches/README.md)

Eight agents searched for methods that could help compute the accuracy of a piecewise-linear ReLU network under
Gaussian inputs. Citations marked unverified in each file were written from memory and need checking. No ARC AlgZoo
material from 2026 was read.

| File | Scope | Ideas that became experiments |
|---|---|---|
| [pre_2000.md](pre_2000.md) | Classic orthant probabilities, tube formulas, Laplace's method, FORM/SORM, order statistics | tube (E15), reliability (E17), sci_laplace (E26) |
| [2000s.md](2000s.md) | Line sampling, Genz-Bretz, Miwa, piecewise-affine region exploration, Array-RQMC | array_rqmc (E16), cone_enum (E14) |
| [2010s.md](2010s.md) | Botev minimax tilting, preintegration + RQMC, region counting, multilevel splitting | cone_enum (E14), tie_lines (E18) |
| [2020_present.md](2020_present.md) | Preintegration theory, combination line sampling, cumulant propagation, LIN-ESS, probabilistic branch-and-bound | tie_lines (E18), branch_bound (E19), propagation (E13) |
| [domain_mathematics.md](domain_mathematics.md) | Braid arrangement, conic geometry, tropical geometry | tube (E15), tie_lines (E18) |
| [domain_neuroscience.md](domain_neuroscience.md) | Threshold-linear networks, ideal-observer analysis, time cells | observer (E21) |
| [domain_physics.md](domain_physics.md) | Perturbation resummation, Rhee-Glynn debiasing, multilevel MC | debiased_series (E20) |
| [domain_engineering_finance_cs.md](domain_engineering_finance_cs.md) | Directional simulation, digital-option smoothing, probabilistic verification | great_circle (E22) |

Experiment IDs refer to the [INDEX registry](../INDEX.md#3-experiment-registry); folders are indexed in
[m16/approaches/README.md](../m16/approaches/README.md).
