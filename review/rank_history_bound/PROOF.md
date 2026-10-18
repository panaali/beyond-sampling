# T245: hidden-image rank bounds next-history complexity

## Statement and scope

Let C be a full-dimensional convex polyhedral cone in R^t representing one complete exact gate history of a bias-free scalar-input ReLU network. On this cone let its current hidden state be h=A x in R^H, with A linear and A(C) contained in the nonnegative orthant. Gates use ON for strictly positive preactivation and OFF otherwise. Count only child words that occur on a set with nonempty interior in R^(t+1); boundary-only words are excluded. A coefficient-identically-zero gate is always OFF, rather than a strict inequality to satisfy.

Let r=rank A. The next preactivations at new scalar input u are

    z=W h+w u.

For r>0 the number of positive-volume next words from this parent is at most

    B(H,r) = sum from j=0 to min(r,H) of binom(H,j).

For r=0 the bound is two if w is nonzero and one if w=0. The r=0 case must not be obtained by substituting r=0 in B. No assumption of generic weights, distinct hyperplanes or nonzero individual input coefficients is needed. The bound counts gate histories, not minimal affine maps, labels, Gaussian masses or necessary algorithmic operations.

## 1. The effective image has an open positive section

Write D=int C and L=im A, a vector space of dimension r. Because D is nonempty and open, its span is R^t and span A(C)=L. The linear map A:R^t→L is surjective and open: a complement to ker A gives an invertible restriction onto L, so the image of every sufficiently small input neighborhood contains an L-neighborhood. Consequently J=A(D) is nonempty, open relative to L, convex, and closed under strictly positive scaling.

For r>0, zero cannot belong to J. Otherwise its relative openness would supply a neighborhood of zero in a nonzero L, hence a nonzero vector together with its negative in J; this contradicts J⊂R_+^H. Thus every h in J satisfies

    ell(h)=sum_i h_i > 0.

The affine section S={h in J:ell(h)=1} is nonempty, convex, and relatively open of dimension r−1. Nonemptiness follows by positive scaling, and ell restricted to L is a nonzero functional. Every (h,u) in J×R can be written uniquely as

    (h,u)=ell(h) * (h/ell(h),u/ell(h)),

where the normalized pair belongs to S×R. This domain is convex and relatively open of dimension r. Multiplication by the positive factor preserves every gate's sign, including zero ownership.

The proof needs no pointedness of C itself: directions in ker A may be discarded. The nonnegative hidden image supplies the positive section. The statement that A(C) spans L uses full dimension of C; a lower-dimensional restriction needs its own relative-domain formulation with dim span A(C), not an unexamined identification with rank A.

## 2. Open children correspond to arrangement chambers

The map (x,u)→(A x,u) sends D×R onto J×R. Each new preactivation is linear there; after normalization it is an affine function on S×R. A gate whose function vanishes identically on this domain is fixed OFF and imposes no hyperplane. A nonzero constant has a fixed sign. Other gates induce at most H proper affine hyperplanes in the r-dimensional section space.

Any child with nonempty ambient interior has a point in D×R avoiding the finitely many zero sets of nonzero new forms. Indeed those proper hyperplanes have empty interior, as does the parent boundary. Thus it has a strict sign word for every non-identically-zero form. Normalization sends that point into an open arrangement chamber on S×R.

Conversely, a nonempty open arrangement chamber in S×R gives a nonempty open sign region in D×R: lift a section point to some x in D, and use the strict finite sign inequalities plus openness of D to obtain a neighborhood. Hence a realized chamber cannot be merely a boundary child. Different strict gate words give different chambers. Each sign region is an intersection of affine halfspaces with a convex domain, so it is convex and cannot split into multiple disconnected pieces with the same word.

These observations also handle globally coincident/oppositely oriented hyperplanes and identically-zero forms. One does not count all lower-dimensional faces of the arrangement as additional histories.

## 3. Arrangement count and aggregate bound

H affine hyperplanes in r dimensions cut space into at most B(H,r) open chambers. A standard induction inserts the last hyperplane: old chambers persist, while the number split is no larger than the chamber count induced by the earlier hyperplanes on that (r−1)-dimensional plane. The recurrence R(H,r)≤R(H−1,r)+R(H−1,r−1), with R(0,r)=R(H,0)=1, gives the binomial sum. Degenerate, parallel, constant or coincident forms only reduce the count. Convex restriction to S×R cannot increase it.

For r=0, h=0 on the whole parent. The new forms are w_i u. If w is nonzero, u>0 and u<0 give at most two positive-volume words; if w=0, all gates are always OFF and there is one. The set u=0 may have an additional all-OFF word, but has zero (t+1)-dimensional volume and is excluded by definition.

For a complete collection of exact full-dimensional parents P_t, with ranks r_C, the deterministic aggregate consequence is

    N_(t+1) <= sum over C in P_t of b(H,r_C),
    b(H,r)=B(H,r) for r>0; b(H,0)=2 if w!=0, else1.

If w!=0, the r=0 term is bounded by B(H,t) for t>=1, and every r>0 satisfies r<=min(t,H), so this is no weaker than T236's dimension-only per-parent bound. It may be equal to that bound on every parent. This formula uses an actual complete parent collection and exact ranks; it supplies no observed rank distribution or cost-free way to obtain one.

## 4. Invariances and interpretation

A consistent hidden permutation or invertible positive-diagonal state rescaling changes A by an invertible left map, hence preserves rank. An invertible time-coordinate substitution changes it by an invertible right map, and a positive overall scalar changes neither rank nor gate-history cardinality. Thus T236's positive global weight rescaling correspondence preserves ranks of corresponding prefix maps as well as counts. These algebraic invariances do not preserve Gaussian masses under arbitrary time rescaling.

Lower exact rank can permit a smaller upper bound. It does not force an actual count reduction, establish an explanatory difference between trained and random networks, or show cheap probability integration. Full-rank images can have few realized chambers because their domain misses intersections; rank-one images can realize H+1 words. Distinct parent histories may have equal states but are still distinct complete histories in this count. Output-label equivalence and aggregation across parents are separate methods.

The theorem assumes complete exact parent domains and exact state maps. Numerical ray sets, floating rank thresholds, recipe approximations or a finite witness list do not establish those premises. No source's trained counts, mass coverage, future growth factor, k10 complexity or architecture capacity conclusion is certified here.
