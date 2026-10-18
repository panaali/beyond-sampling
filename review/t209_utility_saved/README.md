# Question
Does the fixed64 cap utility report match its saved rays, exact bounds and accounting?

# Method
`check_saved.py` independently reconstructs the rational cap formula using our previously reviewed T200 reviewer helper, validates original96 archive identities and64 exact ray bindings, reuses published KKT certificates, contracts the determinant upper bound and independently sorts/sums all diagnostics. No author imports or cap/mass/geometry calls. `pins.json` freezes the source and raw inputs.

# Status
Bounded saved-only independent audit complete. Source clearance b443c326 and immutable execution record were published2e8c5f2e before the sole run.

# Results
All64 rows,96 original archives,130 progress records, exact bounds/outward bits/ledgers and threshold contractions pass. At thresholds1e-4 through1e-8, cap counts are36/26/16/5/1 versus determinant64/62/59/50/33. Median outward-bound/stored-reference ratios are6655.295 versus1.0978136; these are descriptive against the conditional reference, not rigorous tightness intervals. No conditional-reference disagreements or cap fallbacks. Alternatives tested are wrong ray/certificate bindings, inward conversion, threshold/median contraction errors and missing inherited certificate costs. The existing independently reviewed exact KKT proof is reused, not solved again.

# Compute accounting
One20CPU/2GB review-lock run, one thread; reviewer process CPU0.114123771s excludes Git-child CPU (external Bash timing preserved). Author phase CPU0.750322521s remains separate. Zero model/mass/reference/MC/candidate calls. Exact rational arithmetic and archive reads are review work; author cost remains separately recorded. Preserve failures and no automatic retries.

# Claims
[measured] The audit validates only the fixed represented-hull bounds and saved diagnostic arithmetic, not global pruning utility or production cost.

# Alternatives not ruled out
Other axes, different enclosures, combined bounds, other cones and a full pruning budget remain open. The conditional floating reference is not a full exact geometry/floating certificate.

# Failure modes
Copied ray provenance, conditional reference assumptions, arbitrary-precision resource growth and certificate premises remain explicit. The64 deterministic indices do not provide a population claim.

Execution: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock bash -c 'ulimit -t 20; time -p /workspace/clean-neurons/.venv/bin/python check_saved.py' > run1.log 2>&1`, with outer `ulimit -v 2500000`.

Pre-run review corrected an ancestry endpoint and distinguished research-commit ancestry from shared-publication chronology. No failed audit execution or altered numerical predicate occurred. Wall-clock publication-before-run relies on the preserved author execution record. KKT construction/validation costs are retained; exact arithmetic macro counts are not native time or a full estimator cost.
