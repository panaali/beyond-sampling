# T105 bounded output-forgetting handoff

Public H2T10 seed0, checkpoint SHA256 `004cc1af2e13b635110488da191ac3d89b32a086720225f4805200c05edb4fa1`.

Eight explicitly disjoint input-sign/ratio events have output labels independent of every prior nonnegative hidden state. Exact binary-rational coefficient, supremum, piecewise-affine readout and projective controls establish the event implications; the uniform-output claim does not assert hidden-state reset. Earlier input values remain in the exact second-rank law. In particular, a negative selected input has a nonzero correct contribution and requires its own target-sign sector.

Measured positive Gaussian quadrature:
- event coverage `.3222697928466594`;
- correct joint contribution `.05304445500911171`;
- unresolved complement `.6777302071533406`;
- fresh complete integration `.0212–.0428` CPU seconds, with fresh rules;
- setup/model loading/exact diagnostic controls separate; no portable total-cost or sampling-win claim.

Latest author implementation/results: `fcff464`. T55 independently reviewed all forcing, readout-kink, signed-pair and negative-target formulas. Different-code adaptive scalar reproduction verified the initial positive contribution and the new mixed-prefix positive-D contribution. Final negative-source/scalar review also passed: independent math.erfc plus adaptive integration gives `.00010442263405448726`, 1.79e-17 from the fixed rule; exact zero-threshold T4/T6/T10 controls are1/48,1/320,1/9216. T55 is committing the written artifact; DRI classification remains separate.

Reproduction commands use `/workspace/clean-neurons/.venv/bin/python` from the repository with one thread and the shared compute lock:

```
python m16/approaches/output_forgetting/negative_controls.py
python m16/approaches/output_forgetting/negative_pilot.py
```

The pilot performs fresh integration of all eight events; it does not substitute saved contributions. All prior controls, numerical-coordinate failures and the import collision/repair remain in README and result files. Gaussian tails have explicit conservative omission bounds; quadrature and floating roundoff are not certified merely by convergence.

Open alternatives: the unclassified complement, other initial-state bounds, other event decompositions, full estimator integration, audited same-unit cost, noisy-model validation and H16 transfer. These are untested, not ruled out. Thresholds are sharp only for the stated interval/recession enclosure where proved, not a globally maximal reachable forgetting event. No further event expansion in this bounded phase; worker moves to another eligible task after independent handoff.
