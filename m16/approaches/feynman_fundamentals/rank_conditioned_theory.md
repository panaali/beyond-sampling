# Fixed-target rank conditioning: remove the moving target coordinate

Let X_0,...,X_(T-1) be iid with continuous CDF F, and let Y be the index of
the second-largest input. The original question is

    accuracy = sum_j P(prediction=j, Y=j)
             = (1/T) sum_j E[P(prediction=j | Y=j, q)],
    q=F(X_j) | Y=j ~ Beta(T-1,2).

The conditional q density is T(T-1) q^(T-2)(1-q). Indeed, at a fixed target
value u exactly one of the T-1 other inputs must exceed u. The unnormalized
joint density for Y=j and q is (T-1)(1-q)q^(T-2); its integral is1/T.
The uniform target index does not imply uniform predictions: every target j
still requires the actual network, its chronological position and its readout.

## Exact online rank transitions, including conditioning on the history

Given q and Y=j, choose the unique greater index uniformly from the T-1
non-target indices. At that index draw X from F conditioned on X>u; at every
other non-target index draw independently from F conditioned on X<u.

After observing ANY full processed history, if no greater value has occurred,
each of the m remaining non-target positions still has the same conditional
likelihood. The processed values have identical below-target densities for
each possible remaining greater index. Therefore the greater index remains
uniform over those m slots, independently of the actual values and hidden
state. This proves the transition probabilities1/m and(m-1)/m conditionally,
not just as unconditional marginals. If the greater input has already occurred,
all future non-target inputs are below. Observing the deterministic hidden
state adds no information beyond the full history, so the result also holds
after marginalizing histories that share that state and exceedance flag.

At a non-target step, writing G(h,x)=ReLU(Wh*h+Wi*x):

    V(h,need=1) = (m-1)/m * E_low V_next(G(h,X),1)
                  + 1/m * E_high V_next(G(h,X),0),
    V(h,need=0) = E_low V_next(G(h,X),0).

The count m decreases by one. A target insertion applies G(h,u) deterministically
and does not change m or the need flag. At m=1, need1 forces the final non-target
input above; need0 forces it below. At m=0, only need0 is reachable.

Both conditional expectations can use a uniform scalar integration variable v:

    X_low = F^-1(q*v),
    X_high = F^-1(q+(1-q)*v),       0<v<1.

For each fixed q this is a two-dimensional hidden-state transport when H=2.
The target value never moves or gets interpolated. The remaining numerical
outer q integration is one-dimensional, naturally using Gauss-Jacobi nodes:
on z in[-1,1], parameters alpha=1,beta=T-2, q=(z+1)/2. Normalize the positive
weights to sum1. This exact measure reduction is distinct from the previous
normalized-radius three-dimensional collocation and its repeated b interpolation.
It does not by itself solve hidden-state interpolation error or quadrature cost.

## A finite domain with an honest missing-mass interval

For standard Gaussian inputs, condition the ORIGINAL iid input vector on the
box |X_i|<=L for every i. The coordinates remain iid, each with truncated CDF

    F_L(x) = [Phi(x)-alpha]/Z, alpha=Phi(-L), Z=1-2*alpha.

All rank and Beta formulas above remain exact for this common continuous CDF.
The inverse maps are finite:

    u = Phi^-1(alpha+Z*q),
    X_low = Phi^-1(alpha+Z*q*v),
    X_high = Phi^-1(alpha+Z*(q+(1-q)*v)).

If A_box is the conditional network accuracy, the unconditional accuracy lies in

    [Z^T*A_box, Z^T*A_box+1-Z^T].

Using A_box itself as an estimate has absolute truncation error at most1-Z^T.
This is a bound on discarded input probability, not a bound on collocation or
quadrature error. Compute the small mass stably as -expm1(T*log1p(-2*alpha)).
At T10, L5 loses about5.73e-6 and L6 about1.97e-8. A certified finite hidden
box follows from B_0=0, B_(t+1)=max(Wh,0)B_t+abs(Wi)*L. Such a box can be loose;
better grids and convergence checks remain necessary.

## Cheap exact final scalar-input kernel

With one non-target input x left, represent every hidden coordinate as an
affine function of x on intervals. Split intervals at ReLU roots. If j is the
last chronological position, perform the fixed-target insertion afterwards,
splitting again where its affine preactivations change sign. Finally intersect
the affine inequalities requiring readout j to win. For positive-length cells
with identically equal logits, preserve first-index argmax. Single crossing
points have zero Gaussian mass, but identical affine rows do not.

Intersect accepted intervals with [-L,u] or [u,L], according to the need flag.
Their Gaussian CDF mass is divided by Z*q or Z*(1-q), respectively. Positive
tail differences should use Phi(-lo)-Phi(-hi), avoiding subtraction from1.
For H2 there are at most16 fresh-plus-target gate patterns. This kernel requires
no two-dimensional Gaussian polygon integration. T67's owner implements it;
T55 provides this derivation only.

## Exact toy controls before any trained claims

A constant readout gives accuracy1/T for every continuous iid input law, and
the conditional-rank transition weights preserve this exactly when quadrature
weights are normalized. For H1 with Wi>0 and Wh=0, let the network choose class
J on a positive final input and class0 otherwise. If J is not the final index,
accuracy is1/T (including J=0). If J=T-1, accuracy is

    1/T + [1/2-(T+1)/2^T]/(T-1).

This holds for every symmetric continuous input law, including the common
truncated Gaussian. To check the new measure independently, integrate against
the Beta density. The target-last hit probability is1[q>1/2]. For a different
target, the conditional probability that the final input is nonpositive is

    L(q) = (T-2)/(T-1) + (1/2-q)/[(T-1)(1-q)]       if q<=1/2,
           (T-2)/(T-1) * (1/2)/q                    if q>1/2.

The class0 contribution plus the target-last contribution, divided by T,
recovers the formula. For a nonfinal J, contributions L(q) and1-L(q) cancel
to1/T. These tests check rank normalization and chronological handling, but
do not exercise a general recurrent hidden-state interpolation engine.
