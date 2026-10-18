# Fixed exact trained-weight predicates

Claim8dc60612 precedes implementation; controls98b661f7 independently9a2f2bb9. Three fixed models only: H4T3seed0, H4T4seed0, H8T4seed0, all k30. Reuse audited exact model archives (3ee0667b/3bb92013), with original checkpoint/archive hashes. No torch load, network input, geometry/capture, mass, reference integration or MC call. Compute16/16/256 subset sums and rational rank once; retain each predicate failure as inconclusive. No selection by observed predicate result.

Rank uses reviewed exactRREF; subset recurrence enumerates every mask, including empty/full. Model-dependent Fractions/GCD/decoding and named arithmetic are real work, not free preprocessing; native/control/hash/temp-bit costs remain unpriced. CPU timers may be below resolution. Independent saved audit required before any theorem applicability claim is registered.

Alternatives: global margin/projective identity is stronger than future-label identity or domain-restricted agreement. Even a successful weight certificate leaves those, approximate merging and identical-state resets open. Failure to satisfy the certificate neither proves nor disproves identifiability. More widths/seeds/horizons require fresh scope.
