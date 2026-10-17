# Installed fixed-call DSTERF exclusive numeric envelope

Candidate pending independent source/binary review. This extends the separately reviewed source loop bound; it does not replace the original240,946 source result or include child costs.

Installed11ae69a..11ae6ae forms30N as2*(16N-N). The QR/QL shift checks and increments are at11ae999..11ae9c0 and11aed9c..11aedd0; neither path advances beyond the global cap. The recurrence and deflation loop structures correspond to the source proof: at most30N shifts,32N search visits, and N blocks/terminal visits. Only the fixed successful twelve-call capture is composed; its six DSTERF calls have no DLASCL child, and original arrays are retained. No scaling branch cost is silently assigned zero for other inputs.

The body is scalar SSE2 except masks and pairwise entry squaring. Charge two lanes for every ANDPD/ANDNPD/ORPD numeric mask, even though only one result lane matters. Zeroing PXOR is initialization, not numeric work. Arithmetic/compare/sqrt scalar instructions cost one; index/control/memory remains outside this category.

* Initial split scan11ae705..11ae740: three2-lane abs masks, two scalar sqrt, two multiplications, one compare:11 per visited entry.
* Iteration search11ae940..11ae968 (and mirrored QL path): two2-lane abs masks, two multiplications, one compare:7 per entry.
* Shift SIGN composition11aea3b/11aea5c/11aea6c and mirrored11aee... uses three2-lane mask operations,6 in place of the source's one named SIGN. Thus the source12 outside-recurrence units become17 per shift.
* Both recurrence bodies11aeae3..11aeb63 and11aeef6..11aef7e retain13 scalar arithmetic/compare operations in the worst branch. Their optional E assignment and alternative C==0 branch only reduce this bound.
* Endpoint choice has two2-lane abs masks and one comparison; including the three norm guards gives8 per outer block. Initial constant arithmetic/sqrt remains6. Entry-squaring vectorization has no discarded lanes and costs at most one product per entry. Terminal sqrt and failure scan retain2N conservative units.

Therefore the exclusive installed numeric-category envelope is

`12N(N-1) + 224N(N-1) + 390N(N-1) + 510N + 8N + 2N + 6`

`= 626N(N-1) + 520N + 6`.

For orders6,3,3,6,10,16 this sums to274,568. All named children remain separate; the fixed trace records85 DLAPY2,6 DLAE2,6 DLANST and6 DLASRT calls, while DLAMCH returns precompiled constants with zero floating arithmetic under the independently reviewed helper convention. Character/runtime work is not free hardware work. No fullSVD, utility, reduction, integer, memory or complete estimator claim follows.
