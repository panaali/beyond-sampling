# T67 target-measure coordinate substitution

Read-only source review of `measure_cycle/driver.py`4053bfc and completed
pilot d8033df. The truncated input CDF is q=(Phi(u)-Phi(-6))/Z, with
Z=1-2Phi(-6). Second-largest target density in q is
T(T-1)q^(T-2)(1-q); multiplication by phi(u)/Z is the correct u Jacobian.
The per-label1/T factor is preserved by both implementations. Neither
coordinate renormalizes omitted target mass. Tiny-panel omitted widths
are multiplied by a valid global upper bound on the target density.
Input-support omission remains separately reported.

`check_recovered_precision_review.py` independently sums saved per-label
contributions with60-digit Decimal arithmetic. GL2 toGL4 changes are
1.7716740302e-7 in q and1.7601347581e-7 in u. Both exceed the frozen1e-7
gate. At GL4 the coordinate difference is only-3.311125e-11, while the
reference errors remain-.00351887824793264 and-.00351887828104389.
The saved double sums differ from the Decimal sums by at most2e-17.

This falsifies the proposed coordinate substitution as a sufficient
repair of that particular stability gate. It does not establish failure
of all adaptive rules, complete topology, or the transport family. The
spatial/model representation bias is much larger than this discrepancy.
No fresh full recurrence is repeated for this audit.
