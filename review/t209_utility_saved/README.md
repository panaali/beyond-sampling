# Question
Does the fixed64 cap utility report match its saved rays, exact bounds and accounting?

# Method
`check_saved.py` independently reconstructs the rational cap formula using our previously reviewed T200 reviewer helper, validates original96 archive identities and64 exact ray bindings, reuses published KKT certificates, contracts the determinant upper bound and independently sorts/sums all diagnostics. No author imports or cap/mass/geometry calls. `pins.json` freezes the source and raw inputs.

# Status
Source frozen; awaiting independent source gate before one saved-only run.

# Results
Pending. Alternatives tested are wrong ray/certificate bindings, inward conversion, threshold/median contraction errors and missing inherited certificate costs. The existing independently reviewed exact KKT proof is reused, not solved again.

# Compute accounting
One planned20CPU/2GB review-lock run, one thread. Zero model/mass/reference/MC/candidate calls. Exact rational arithmetic and archive reads are review work; author cost remains separately recorded. Preserve failures and no automatic retries.

# Claims
[inferred] Passing this audit would validate only the fixed represented-hull bounds and saved diagnostic arithmetic, not global pruning utility or production cost.

# Alternatives not ruled out
Other axes, different enclosures, combined bounds, other cones and a full pruning budget remain open. The conditional floating reference is not a full exact geometry/floating certificate.

# Failure modes
Copied ray provenance, conditional reference assumptions, arbitrary-precision resource growth and certificate premises remain explicit. The64 deterministic indices do not provide a population claim.
