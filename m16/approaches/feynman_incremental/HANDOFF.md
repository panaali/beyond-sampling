# T20 bounded-phase handoff

Status: Feynman and uniform radial tube ports implemented and numerically checked; DRI acceptance/full source review pending. Existing Gaussian primitives preserved. No full-challenge accuracy or sampling advantage claim. Remaining Laplace, boundary, fitted-importance/Newton/hybrid variants remain open.

## Results
- Feynman trained checkpoint512/2048 parity to1.11e-16 on identical node/face hashes; cold proxies241372129/252973153. Cold setup200000500 is explicit. Scalar tail and90 independent line fixtures pass. Independent constant-label formula isolates1.7627pp coarse outer-rule bias.
- Uniform radial tube trained checkpoint512/2048 parity<=9.66e-13; cold proxies118329903/133088609. Radial scalar28 and rational54 controls pass, plus finite-domain/cap checks. Positive70-digit constant-label formula independently diagnoses the legacy zero-margin tie defect; corrected outer rule still has5.50pp bias at512.
- All costs are explicit numerical proxies; Sobol/RNG/special functions are opaque conventions, not portable bounds. Numerical roundoff and geometry-table inverse interpolation are not certified. No Monte Carlo/noise scoring performed.

## Main files
`estimator.py` and `tube_estimator.py` expose est(model,budget,rng); detailed methods return ledgers. `native.py` and `radial.py` adapt only interval integration around pinned shared walker logic. README.md and RADIAL.md contain frozen alternatives, precise commands, failures and limitations. Saved JSON contains full operation categories and node fingerprints. Constant-output diagnostic runners perform independent positive integration rather than trusting legacy outputs.

## Reviews
T55 reviewed midpoint correction, radial variable change, beta branches, derivative bound and coarea normalization. Its finite-endpoint caveat was fixed and tested. T30 owner independently confirmed constant-label truth1/T and requested precise bias isolation; the resulting independent diagnostic is saved. Root/DRI should review full adapters before classification; author does not self-confirm.

## Next phase and publication blocker
T20 radial scope was pushed2a9c932. Prepared next TASKS-only extension57e235d lives in `/workspace/work/admin-small34-laplace`, based on latest8f271c3. Push failed with missing GitHub username after runtime credentials disappeared; current environment status has no credential bindings and git has no helper. Normal proxy/escalation was preserved, no credential workaround. Laplace implementation has not started. Its source uses exact near radial mass plus one-point tail evaluation; next phase must preserve/charge the tail and report its full mass as uncertainty, not zero remainder.

## Complete research chain since last root-integrated277dc01
Cherry-pick in this order; all commits edit only this owned folder. The separate TASKS commits above are already shared or explicitly pending and must not be cherry-picked as research.

```
b68111d Freeze whole Feynman port and explicit cold-cost proxy
6aa6d14 T20 freeze stable normal-tail interval checks
524a800 T20 use reflected tails and locally bounded midpoint intervals
a1f27b1 T20 record reflected-tail and native fixture validation
7680150 T20 add cold counted whole-estimator adapter
47e3d26 T20 freeze paired-node whole-port validation runner
f96f54e T20 resolve restored checkpoint by exact hash
4e841c0 T20 record full port parity costs and failed constant-output accuracy control
3dd4788 T20 freeze failure taxonomy and cold warm interface controls
9030852 T20 preserve unexpected kernel failure taxonomy and partial charges
a7ac0ab T20 record cold warm and partial failure interface checks
d7caa7a T20 freeze independent constant-label bias isolation
dd51dca T20 independently isolate constant-label outer-rule bias
c52f65c T20 retain face fingerprints in independent diagnostic callback
2965999 T20 record independent constant-label face and normalizer checks
364be04 T20 freeze radial primitive normalization and independent falsifiers
364bd0c T20 implement radial beta interval callback and independent fixtures
b00b200 T20 resolve wide-interval positive reference at fixed physical breakpoints
3d170d8 T20 record independently integrated radial primitive controls
8eed02c T20 freeze uniform-face radial tube full-port comparison
877e6df T20 add uniform radial tube adapter with cold cost ledger
b559882 T20 enforce finite radial helper endpoints after independent review
5a293e8 T20 freeze independent radial constant-label tie diagnosis
f61cbd6 T20 independently reconstruct radial constant-label face masses
5b143f3 T20 record full radial port costs and independently diagnosed legacy tie defect
```
