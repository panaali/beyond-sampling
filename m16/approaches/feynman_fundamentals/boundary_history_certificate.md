# Two-input H2 boundary model and omitted-history certificate

T55 derivation supporting T76, 2026-10-17. T76 owns the implementation. This
construction uses a weight-derived two-input surrogate; it does not assume the
real model is nearly perfect or that its errors are thin wedges. The confirmed
T63/R2 mechanism result motivates boundary descriptions but does not establish
their transfer across models, dimensions or noise schemes.

## Exact rank weights for a surrogate that forgets the prefix

There are m=T-2 omitted iid standard-normal inputs, followed by x,y. Start the
surrogate hidden state at zero before x,y. Its output depends only on those
last two inputs. Put U=max(x,y), L=min(x,y), qU=Phi(U), qL=Phi(L). Conditional
on x,y, the probability that its predicted label is the true second largest is:

* predicted larger of the last two: m*(1-qU)*qU^(m-1);
* predicted smaller of the last two: qL^m;
* any fixed omitted label, for m>=1:
  1/m-qU^(m-1)+qU^m-qL^m/m.

For an omitted label with value u, there are two possible cases. If u>U,
exactly one of the other m-1 omitted inputs must exceed it. If L<u<U, all
other omitted inputs must be smaller. Integrating their densities gives

    integral_qU^1 (m-1)*(1-q)*q^(m-2) dq
       + integral_qL^qU q^(m-1) dq,

which simplifies to the stated expression (for m=1, the first integral is
zero). Summing the m equal omitted-label probabilities and the last two
probabilities gives one exactly. The integral form also makes nonnegativity
clear; near endpoint probabilities a direct subtraction may lose precision.

These weights are exact for the surrogate. They do not assert that the actual
hidden state is independent of prefix ranks. Instead, for any shared true label,

    |accuracy(real)-accuracy(surrogate)| <= P(real prediction != surrogate prediction).

This elementary coupling is the bridge to an actual-model error certificate.

## Independent radial check for T4/T5

Write (x,y)=R*(cos(theta),sin(theta)), with R Rayleigh and theta uniform and
independent. The two-input surrogate is homogeneous, so its prediction depends
only on theta. Define F_n(a)=E_R[Phi(aR)^n], F_0=1. Integration by parts gives

    F_n(a)=2^(-n)+n*a/sqrt(1+a^2)*J_(n-1)(b),
    b=a/sqrt(1+a^2),
    J_k(b)=integral_0^infinity phi(t)*Phi(b*t)^k dt.

For n<=3 these integrals are Gaussian orthant probabilities. With
c=a/sqrt(1+2*a^2), they are

    J_0=1/2,
    J_1=1/4+asin(c)/(2*pi),
    J_2=1/8+[2*asin(c)+asin(a^2/(1+2*a^2))]/(4*pi).

The signs remain correct for negative a. For J2 the three variables are a
positive reference normal and two independent normals compared with b times
that reference; correlations are c,c,c^2. This independently verifies T76's
formulas and all normalization factors.

Consequently, at each angle with aU=max(cos(theta),sin(theta)) and aL=min(...),
the radially integrated rank weights are

    larger-last: m*(F_(m-1)(aU)-F_m(aU)),
    smaller-last: F_m(aL),
    omitted label: 1/m-F_(m-1)(aU)+F_m(aU)-F_m(aL)/m.

For T4/T5, m=2/3, so only the displayed elementary formulas are needed. The
remaining scalar angular integration must still be charged and controlled.
Its sectors come from the two-step gate and H2 readout boundaries, not the
full model's gate history. No accuracy reference enters these coefficients.

## An independent prefix-box certificate

Fix a box cutoff B>0 before examining errors. On the event all m prefix inputs
lie in [-B,B], probability p_box=(2*Phi(B)-1)^m, bound the omitted hidden state
componentwise using its nonnegativity. Let Wplus=max(W,0), Wminus=max(-W,0).
Starting H0=0, use

    H_(k+1)=Wplus*H_k+B*|w|.

Then 0<=h_prefix<=H_m, since ReLU(W*h+w*x)<=Wplus*h+|w|*|x|. This is tighter
than using |W| throughout the prefix, especially for inhibitory recurrent rows.

For the first retained input, the actual preactivation shift W*h_prefix lies
between -Wminus*H_m and Wplus*H_m. ReLU is1-Lipschitz componentwise, so the first
retained hidden-state difference is at most

    e1=max(Wminus*H_m, Wplus*H_m)

componentwise. Before and after the final retained ReLU, the error is bounded by

    d=|W|*e1.

This bound is independent of the last two inputs and their angle/radius. A
looser but valid alternative is B*sum_(r=0)^(m-1) |W|^(r+2)*|w|. Inhibitory
weights can make that looser bound uninformative; testing the tighter box bound
before concluding failure costs only short matrix-vector recurrences.

## Margins and positive-mass tie strata

At fixed angle, write the surrogate final hidden state as R*b(theta), and its
final preactivation as R*a(theta). Let j be its first-index argmax label.
For any competing row k, the output-margin perturbation is at most
e_jk=|Wo_j-Wo_k| dot d. If every relevant surrogate margin
g_jk=(Wo_j-Wo_k) dot b(theta) is strictly positive, then

    R > max_k e_jk/g_jk

certifies the same prediction. Globally identical readout rows can be handled
by exact first-index ownership. Nonidentical zero margins cannot simply be
ignored: they may carry positive probability on lower-dimensional supports.

There is a stronger cheap certificate on dead and single-axis sectors:

* If both a_i<0, R>=max_i d_i/(-a_i) ensures both actual final preactivations
  remain nonpositive. Both networks therefore output zero hidden state and
  predict class0, including its positive-mass tie.
* If exactly one a_i>0 and the other is negative, R>max_i d_i/|a_i| preserves
  both final gate signs. Both hidden states lie on the same positive coordinate
  axis. Their first-index readout winner depends only on that readout column,
  not amplitude, so all ties on that axis are handled without a positive margin.

If a_i=0 and d_i=0, that gate is identically off and needs no positive threshold.
If a_i=0 but d_i>0, this gate certificate is unavailable. A surrogate input-ray
tie with both gates active is not automatically protected by the axis argument.
When several valid certificates apply, take the smallest required radius.

Let r_*(theta) be any resulting sufficient radius, infinity when uncertified.
Independence of prefix and retained inputs and the Rayleigh survival function
give the agreement lower bound

    P(agreement) >= p_box/(2*pi) integral_0^(2*pi) exp(-r_*(theta)^2/2) dtheta.

Its complement bounds the accuracy difference. The angular integral must be
bounded from below if used as a rigorous certificate; an unbounded numerical
quadrature estimate is only a diagnostic. Activation/margin interval bounds
on angle cells could provide conservative lower sums without full gate-history
enumeration. The bound can still be vacuous; that is a useful falsifier of this
particular erased-history approximation, not evidence that the network lacks
other low-complexity mechanisms.

## Scope and checks still owned by T76

The envelope must include the origin's class0 and exact first-index axis ties
when reporting reachable labels; nonzero readout support alone can miss them.
The network is bias-free and inputs independent standard normals throughout.
Errors in geometry, floating gates, elementary functions and quadrature need
their own accounting. Validate constant/dead/axis models, exact rank-weight
normalization and negative angular arguments before trained estimates. No
numerical experiment was performed for this derivation memo.
