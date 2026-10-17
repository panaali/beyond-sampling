# T187 fading-memory influence diagnostic

## Question
Do old inputs have smaller certified output-margin sensitivity, and is a cheap
weight-only bound informative enough to justify a subsequent merging test?

## Method
Exact common-dyadic integer recurrence and local affine derivatives in core.py;
mask-independent absolute recurrent-product bounds, all output pairs. First12
analytic/domain/invariance controls in controls.py, with exact independent
rational forward checks inside each certified radius. Then separately frozen
H4T4/H8T4 seed0, same384 signed permutations of(1,2,3,4) per model. These are
finite local witnesses, never a sampled probability or accuracy estimate.
Related T49 reset events/T105 suffix forgetting/T108 cumulative bounds do not
replace the present strict-region checks. No geometry enumeration or merging.

## Status
Claim ba0bc420 pushed before implementation. T55 source gate passed; all12 exact controls pass in0.063825CPU, with80 rational interior perturbation checks. Saved independent audit pending; no trained evaluations have run.

## Results
Controls complete; no trained result. Commands: python controls.py, later separately frozen trained runner.
Existing output files are immutable; failures and partial progress retained.

## Compute accounting
Named exact integer add/multiply/subtract counts and retained result bitwidth;
separate rational control forwards; imports/setup, process CPU, memory and pins.
Comparisons/signs/max/abs, conversion/gcd and native/control/memory costs remain
outside named arithmetic. Model evaluations are charged even at deterministic
inputs. This is a diagnostic, not an estimator cost or sampling comparison.

## Claims
[inferred] The global bound and strict-region derivative follow PROOF.md.
[speculative] Some margin-aware state merges may be possible; no evidence yet.

## Alternatives not ruled out
A large upper bound may be loose: exact local witnesses distinguish this from
actual dependence. A finite witness panel may miss reset/damped cells: no mass
or universal claim follows. Reachability and margin-conditioned bounds remain
plausible even if uniform fading fails. Four-step behavior need not extrapolate.

## Failure modes
Boundary directions retained as unresolved; identically-zero forms separate.
No spectral-radius-only certificate. No Gaussian input truncation, implicit
margin threshold, silent output normalization or approximate-zero gate rule.
One20CPU control phase, one110CPU trained phase after source gate,2GB/one thread.
