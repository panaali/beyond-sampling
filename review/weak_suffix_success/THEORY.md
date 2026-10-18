# Conditional success as a weaker suffix object

Written preparation only. No numerical claim or implementation clearance.

## State and law

Consider the fixed bias-free scalar-input recurrent network h_{s+1}=ReLU(Wh h_s+Wi X_s) and final logits Wo h_T, with first-index argmax. Input coordinates are iid standard Gaussian. At time t>=2, after prefix coordinates0..t-1, define R_t=(a,b,i_a,i_b), where a>b are the two largest observed values and i_a,i_b their indices. Strict prefix ranks hold almost surely; ties and t<2 require a separately specified rank convention and are outside this first contract.

Let S_t=(h_t,R_t). Given the exact prefix, the remaining coordinates are independent standard Gaussians, independent of S_t. Define

q_t(h,r)=E[1{network output from h and the remaining inputs equals the final second-largest index computed from r and those inputs}].

This is a transition kernel defined on supplied h>=0 and strict r, whether or not every supplied state is reachable. On actual prefixes, it is a version of E[correctness | S_t], because the recurrence and rank update use only S_t and independent future inputs. Thus accuracy is E_{mu_t}[q_t(S_t)], where mu_t is the JOINT pushforward law of hidden and rank state from the same prefix. Replacing mu_t by separate marginals or an independent product is unjustified. For example, a generic kernel1{hidden=rank} has expectation1 or0 under two couplings with identical Bernoulli marginals; this illustration is not an asserted trained-network counterexample.

The backward identity is q_t(h,r)=integral phi(y) q_{t+1}(ReLU(Wh h+Wi y),update_t(r,y))dy. At t=T it is the first-index correctness indicator. This identity is not itself an efficient representation: query construction, rank routing and the integral remain work.

## Exact last-input formula in real arithmetic

Fix t=T-1 and supplied state h,r=(a,b,i_a,i_b), a>b. Put c=Wh h and d=Wi. Gates change only at finite roots -c_k/d_k for d_k!=0. A zero slope has constant sign; an identically zero preactivation contributes zero without duplicate gate branches. There are at most H distinct gate roots and H+1 open gate cells.

Within each gate cell, logits are affine in the last input y. Insert all roots of nonconstant pairwise logit differences that lie in that cell. Identically zero differences remain tied on the entire cell and use first-index ownership; do not remove such ties as zero-probability events. Add the rank breakpoints b and a. Then network output and final true second index are constant on each resulting open interval. The true label is i_b for y<b, T-1 for b<y<a, and i_a for y>a. Finite isolated breakpoints have zero Gaussian mass, so their pointwise rank tie convention does not affect this integral. Permanent readout ties do affect it and remain explicitly owned.

For each interval (L,U), choose any interior representative solely to determine its affine signs/first-index winner and true label. Sum Phi(U)-Phi(L) for matching labels. Infinite endpoints use Phi(-infinity)=0 and Phi(infinity)=1. This is an exact real-arithmetic identity; ordinary floating CDF evaluation, root sorting or near-tie tests are not automatically certified.

A simple upper bound on the number of distinct finite breakpoints is H+2+(H+1)*T*(T-1)/2. It is a bound on this explicit construction, not a lower bound or a complete runtime price. Form c, each gate-cell readout, pair differences, roots, ordering, ownership and CDF endpoints with their actual model-dependent costs. Exact rational roots are possible for rational weights/state, but bit growth, comparisons and sorting are real costs. A directed CDF/tail implementation for arbitrary roots would need its own source/domain controls; old bounded-grid CDF checks do not transfer automatically.

## Weak equality, approximation and reachability

Equal q_t values require neither equal hidden states nor identical predictions for every suffix. A kernel-level exact example uses H2,T3,Wi=0,Wh=identity and Wo rows(1,0),(0,1),(0,0). At supplied h=e1 or e2, predictions are constantly0 or1. With the SAME strict rank state a=A>0 at index0,b=-A at index1, both success probabilities equal Phi(-A), while their correct suffix events are disjoint (y>A versus y<-A). This demonstrates strict weakness of probability equality for supplied states. These states are NOT claimed reachable from the zero start of this Wi=0 network; there is no trained merge or accuracy gain inferred.

There is no automatic uniform continuity of the extended kernel. In the same Wi=0 example with a=2,b=-1, moving h through h1=h2 changes the predicted label and hence q from Phi(-2) to Phi(-1). First-index ownership fixes the value at equality. This is a degeneracy of supplied-state kernel extension, not an assertion of discontinuity on an actual reachable prefix distribution. Existing T132 continuity results impose explicit nonzero input-slope assumptions and are model/domain-specific.

For a weight-derived summary Z=f(S), an estimator qhat(Z) has signed accuracy error E_mu[qhat(f(S))-q(S)]. A valid pointwise bound |qhat(f(S))-q(S)|<=epsilon(S) yields absolute accuracy error <=E_mu epsilon(S), hence also <=sup epsilon. It does not justify replacing the joint law by marginals. If only a finite set of states is checked, its maximum discrepancy is a finite diagnostic, not a bound between those states or a population certificate.

The conditional mean E[q(S)|Z] minimizes L2 error under a specified joint law, but computing that conditional mean and law is the central problem, not free preprocessing. A scalar q table that enumerates every cell to form its entries has not demonstrated cheaper structure. Behavioral disagreement and absolute residual bounds are sufficient controls, not necessary conditions for small integrated signed bias; cancellation must be measured or bounded rather than assumed.

## Preserved limits

T188 exact all-future margin nonmatches do not imply different q values. Conversely, equal or close q values at supplied states do not establish a cheap representation, reachable-state equivalence, source-cell feasibility, appropriate prefix weighting, or generality across weights. One-step integration avoids the second-step chronology and correlated rank flags of older two-input terminals; extending to two future inputs requires explicit joint rank partitioning and separate numerical/error review. No existing frozen negative campaign is restarted here.
