# Physical-coordinate affine enclosure on a fixed separated-gap domain

## Scope and premises

This is a written integration contract under renewed claim1a49af7f. It extends the independently reviewed T256 affine-range primitive (theorya17014a6; controls65ac580f) to the existing T189 enclosure family. No implementation, model decoding, affine recurrence, CDF or new numerical experiment occurs in this phase.

Fix a finite rational physical-coordinate box B, T>=3, a labeled closed domain D=D_ij(B,g) as in the parent PROOF.md, g>=0, and a finite exact-real ReLU RNN h_t=ReLU(W_h h_(t-1)+w_i x_t), h_-1=0, with linear readout W_o h_(T-1). Weights are rational; dimensions/orientation must be validated by any future source. This proof permits additive constants in representations even though this RNN has no bias. ReLU applies componentwise. Domain D remains the SAME fixed input subset through every time step. No conditional independence of the coordinates or propagated errors is assumed.

If D is empty, report empty and do not classify a nonexistent point or attach mass. If D is nonempty but lower dimensional, enclosures are valid geometric statements; no positive probability follows. A closed g0 domain does not establish deterministic second-rank label j at tied points. First-index network-output ties are handled separately below.

## Representation and coordinate ownership

For each hidden coordinate, maintain two simultaneous pointwise statements for every x in D:

h_k(x)=c_k+a_k dot x+e_k(x), with e_k(x) in E_k=[eL_k,eU_k]; and h_k(x) in B_k=[bL_k,bU_k].

Store c,a,E,B as canonical exact Fractions in PHYSICAL x units, with positive denominators. Initial values are all zero. There is no implicit integer common denominator and no binary64 conversion. The existing integer Ledger.see/bit_length cannot receive Fraction values. A future ledger must separately inspect numerator and denominator bit lengths and count rational operations/setup; no constant-cost rational arithmetic or full FLOP equivalence is claimed.

For inherited v=2x, a saved numerator affine expression (c_v+a_v dot v+e_v)/S, S>0, corresponds to c=c_v/S, a=2a_v/S, E=E_v/S; an independent saved range B_v/S becomes B. Equivalently, a T256 range call can keep the numerator objective in v coordinates and replace a physical gap g by2g, then divide its endpoints by S. Constants, remainder numerators and S are NOT doubled by the coordinate substitution. When quantities are already normalized physical Fractions, do not divide again. Readout margins have their own positive saved denominator. These distinctions are explicit regression obligations, not inferred from matching printed gap labels.

## One exact recurrent step

For row w of W_h and input coefficient beta, define

c'=sum_k w_k c_k,
a'=sum_k w_k a_k + beta e_t,
E'=sum_k w_k E_k,

where interval multiplication/addition use exact sign selection, and e_t in the expression for a' denotes the coordinate basis vector, not the error function. Then the preactivation z(x)=c'+a' dot x+epsilon'(x) with epsilon'(x) in E' holds on D. The sum interval does not require independent errors.

Let A_D be the exact T256 affine range of c'+a' dot x on D. Define two independently valid preactivation intervals:

I_aff=A_D+E',
I_direct=sum_k w_k B_k + beta [l_t,u_t].

The latter deliberately uses the original physical box interval for x_t; it remains valid on D and isolates the affine-domain restriction. A later tighter per-coordinate input range is a separate choice, not assumed here. Intersect them to obtain [L,U]. For a nonempty D, L>U is a violated premise/implementation error and must stop; never floor, clip or silently declare D empty. Optional intersection with a separately valid inherited range is also safe, but any chosen integration must freeze which intersections it performs.

For L>=0 choose lambda=1 and r=0. Else if U<=0 choose lambda=0 and r=0. At L=U=0, the first branch owns equality, matching the inherited source convention. If L<0<U choose lambda in {0,1/2,1}. On [L,U],

0 <= ReLU(z)-lambda z <= r(lambda)=max(-lambda L,(1-lambda)U).

Indeed on z<=0 the residual is -lambda z, on z>=0 it is (1-lambda)z, and both meet at zero. This interval contains the entire residual, including its boundary values. The inherited slope heuristic minimizes lambda*(eU'-eL')+r(lambda), breaking ties toward the smaller lambda. It minimizes the reported error-interval width only; it is not asserted to minimize every future margin or the total affine range.

Update

c_new=lambda c', a_new=lambda a',
E_new=lambda E'+[0,r],
B_new=[max(0,L),max(0,U)].

Both pointwise statements hold, including off gates and the zero preactivation. Lambda is nonnegative, so interval endpoint order is preserved. Induction proves these simultaneous affine/error and direct-range enclosures for all times. Dependence discarded by interval arithmetic can make the result loose, but not invalid under these premises.

## Readout margins and old-certificate fallback

For target label j and rival o, take row contrast w=W_o[j,:]-W_o[o,:]. The exact target-minus-rival margin has affine part c_m=sum w_k c_k, a_m=sum w_k a_k, error interval E_m=sum w_k E_k and independent direct interval J_m=sum w_k B_k. Intersect range_D(c_m+a_m dot x)+E_m with J_m. Again, an empty intersection on nonempty D is a failure, not a rounding repair.

An independently valid old whole-box interval O_m also applies to D subset B. Its normalized endpoints must be obtained from the exact saved numerator and POSITIVE margin denominator, with contrast orientation checked. One may report old, new and final intersection F_m=O_m intersect new separately. F_m is contained in O_m by construction and is valid. This guarantees no loss of existing interval/status information after fallback, but DOES NOT prove that the new propagation alone is tighter, that an old unknown is resolved, or that the extra cost is useful. A disjoint intersection exposes an inconsistency and stops the phase. Do not select a candidate based on empirical truth or silently omit the weaker method's cost.

For first-index argmax, j is certified correct when every rival o<j has lower(F_m)>0 and every rival o>j has lower(F_m)>=0. It is certified wrong if some earlier rival has upper(F_m)<=0, or some later rival has upper(F_m)<0. Otherwise status is unknown. These predicates address the supplied target j; at g0 a separate strict-rank/tie policy is needed before identifying j with the task's true label. For g>0, i and j are unique top/second coordinates throughout D.

## Probability and cost boundaries

No Gaussian event mass is computed by this contract. Certifying a partial domain D does not certify its entire parent box. Positive-gap accuracy contributions require P(B intersect E_g) and correct label ownership (T252), or a whole-box inner-cover argument with its own conditions. Feasible witnesses do not supply probability. Physicalgap1 remains untested in the older T240 trained campaign.

Each preactivation/readout affine range invokes a fixed-pair O(T^2) exact-expression primitive; repeated neurons, times, readout contrasts and ordered label pairs all incur their separate factors. Old fallback extraction or recomputation, exact weight conversion, rational arithmetic and numerator/denominator growth must be retained in a future ledger. Neither polynomial expression counts nor finite synthetic timing proves native-cost improvement, sampling competitiveness or a solved rung.
