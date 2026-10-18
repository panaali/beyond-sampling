# T229 independent source gate

Question: is the exact readout predicate a sound sufficient obstruction to nonidentity positive-monomial, unchanged-weight/logit automorphisms?

Status: source/theorem/resource PASS for frozen f64cc107 only. Explicit delegation9b5c91ed. source_gate.json pins all five files and twenty fixed controls. No author module import, predicate/control execution, target weight read or model/RNG call occurred in this review.

A positive-monomial inverse has one positive entry per column. Consequently, Wo A^-1=Wo matches each nonzero readout column to a positive multiple of another. Distinct positive rays force each pairing to be fixed and every diagonal factor to equal one. This sufficient obstruction is independent of the remaining Wi/Wh constraints. Canonicalization retains the first nonzero sign; opposite rays are correctly distinct. Every pair gets a zero-column, positive-ratio, negative-ratio or nonzero-minor witness.

The twelve valid controls contain seven forced-identity and five inconclusive outcomes; eight invalid inputs test explicit exact-coefficient/shape rejection. The fixed near-parallel case distinguishes exact equality from numerical closeness. Twenty attempts and twenty results plus start/complete give42 progress events. Input/results are durable before expectations are asserted. Resource checks and internal16CPU guard require a one-thread, review-lock launcher capped at20CPU and2GiB; any advertised OS soft16/hard20 must be implemented explicitly. The execution gate and this review must be published and hash-bound before the sole run.

Failure modes and alternatives: zero/positive-proportional columns do not establish a true network symmetry; other weights may forbid it. This does not rule out argmax-only/common-logit-shift equivalence, domain-restricted symmetries, quotients, approximate coalescing or other useful representations. No complete native/bit-cost or computational-win claim follows. Synthetic saved certificates require independent audit; target application is a separate frozen scope.
