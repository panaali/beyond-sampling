# T243: conditional residual tails, second moments and a margin split

Proof-only continuation of the independently reviewed T233 identities. Claim
ac8bf56b precedes this artifact. There is no numerical Gaussian evaluator,
target application, code/control run, or claim that the needed margins are cheap.

## 1. Single replacement and conditioning

Fix a finite past state H=h. Let X be a fresh independent standard normal scalar,
and let a=alpha X+c(h), b=beta X+d(h), with finite nonzero slopes of opposite
sign. Write u=-c/alpha, v=-d/beta, L=min(u,v), U=max(u,v), delta=U-L and B=|beta|.
Replace only hidden coordinate j's b+ by the signed surrogate b*1[a<0].
The nonnegative correction is

    r = b+ - b*1[a<0].

Almost surely conditional on h, r=B*|X-v| on L<X<U and zero otherwise.
The exact anchor endpoint X=u can have residual b+; its Gaussian probability
is zero. It must still be retained in pointwise implementations, as in T233's
independently checked controls. In all cases 0<=r<=B*delta, including endpoints.
When u=v, r is identically zero. No Gaussian assumption is made on H.

If conditioning additionally restricts X by ranks, truncation or another event,
the normal-CDF formulas below no longer apply without replacing its conditional
law. The deterministic residual and margin implications remain separate facts.

## 2. Exact conditional tail, including its finite support

Let eta>=0 be fixed conditional on h. If delta=0 or eta>=B*delta, then
P(r>eta|h)=0. Otherwise define

    w_eta = v + sign(u-v)*eta/B.

Within the mismatch interval, r>eta means X is strictly between w_eta and u,
regardless of orientation. Thus

    P(r>eta|h) = |Phi(u)-Phi(w_eta)|, 0<=eta<B*delta.

At eta=0 this gives P(r>0|h)=|Phi(u)-Phi(v)|. At the upper support endpoint
the strict tail is zero; no probability atom has been introduced there. At zero,
there is generally an atom of size 1-|Phi(u)-Phi(v)|. One cannot replace a strict
tail by a non-strict tail at eta=0 without including that atom.

The formula also holds for an H-measurable threshold eta(H), by applying it
conditionally. Choosing eta from X or future realized logits is a different
selection problem; the displayed fixed-threshold CDF expression then does not
automatically give the selected event's probability. No numerical subtraction
of nearby CDF values is claimed stable or certified here.

## 3. Conditional second moment and cubic density bound

Let phi and Phi be the standard normal density and CDF. Direct integration gives

    mu_2(h) = E[r^2|h] = beta^2 * integral_L^U (x-v)^2 phi(x) dx.

With

    F_v(x) = (1+v^2)*Phi(x) + (2v-x)*phi(x),

differentiation gives F'_v(x)=(x-v)^2 phi(x). Therefore

    mu_2(h) = beta^2 * [F_v(U)-F_v(L)].

The exact expression is nonnegative, but its floating evaluation can cancel;
it is a symbolic identity rather than a numerical recipe. Positivity and the
maximum normal density yield two bounds:

    mu_2(h) <= beta^2*delta^2*|Phi(u)-Phi(v)|,
    mu_2(h) <= beta^2*delta^3/(3*sqrt(2*pi)).

The latter follows by integrating the squared distance from endpoint v over
an interval of length delta. It is not a covariance-only bound. If
c(h)=w_i h and d(h)=w_j h, then u-v=ell h for ell=w_j/beta-w_i/alpha.
Provided the directional absolute third moment is finite,

    E[r^2] <= beta^2/(3*sqrt(2*pi)) * E[|ell H|^3].

Knowing only the past covariance does not supply this third moment. If it is
unavailable or infinite, this bound does not become a usable finite quantity.
It also does not make integrating over H inexpensive.

For eta>0, a looser alternative to the exact tail is Markov:

    P(r>eta|h) <= min(1, P(r>0|h), mu_2(h)/eta^2).

The explicit zero tail beyond B*delta should be retained even if this generic
moment bound is positive.

## 4. Small-band asymptotics are not a relative-precision guarantee

Fix finite v and B>0, and approach u=v from either side with delta>0.
Changing variables to distance from v and using continuity of phi gives

    P(r>0|h)/delta -> phi(v),
    E[r|h]/delta^2 -> B*phi(v)/2,
    E[r^2|h]/delta^3 -> B^2*phi(v)/3.

Consequently

    E[r^2|h] / E[r|h]^2 ~ 4/(3*phi(v)*delta),
    Var(r|h) / E[r|h]^2 ~ 4/(3*phi(v)*delta).

The second relation subtracts one from a quantity diverging to infinity.
Thus small absolute residual mean can coexist with large relative variance.
This is a fixed-v local limit, not uniform in tails or varying slopes/states,
and not a claim about the trained model's threshold distribution. It does not
prove that a proposed conditional or deterministic estimator fails.

## 5. Deterministic downstream margin envelope

Run the original and single-surrogate states through the SAME subsequent s
inputs and recurrent weights. No independence among those future inputs is
needed for the following pointwise bound. Coordinatewise ReLU is 1-Lipschitz:

    |h_final - htilde_final| <= |Wh|^s e_j r.

For output rows indexed by y,k, define

    K_yk = |Wo_y-Wo_k| |Wh|^s e_j,
    K = max over distinct y,k of K_yk.

Assume at least two output classes. K is deterministic, nonnegative and finite
for finite fixed weights and s. It bounds EVERY pairwise logit-margin change
by K*r, even though the surrogate intermediate coordinate can be negative.
This is the prior T233/T108 propagation specialized to one replacement, not a
new cheap computation or cancellation-aware envelope.

Let ytilde be the first-index argmax of surrogate logits and let

    G = ztilde_ytilde - max_{k != ytilde} ztilde_k >= 0.

Ties give G=0. If G>K*eta and r<=eta, every original logit margin from ytilde
to another output remains strictly positive. The original winner is therefore
ytilde. With D the disagreement event, the deterministic inclusion is

    D subset {G<=K*eta} union {r>eta}.

For fixed eta conditional on H=h, the union bound gives

    P(D|h) <= min(1, P(G<=K*eta|h) + P(r>eta|h)).

The exact tail in section2 supplies the second term. G and r may be strongly
dependent; the union bound needs no independence. The first term still requires
the joint dynamics and future-input law; it is neither evaluated nor known
cheaply by this proof. A threshold eta(H) measurable with respect to the past
can be used conditionally and integrated over H. Optimizing thresholds from
validation outcomes needs its own selection/accounting argument.

If K=0, all pairwise margins are unchanged, so D is empty including ties under
the same first-index rule. This stronger special case avoids dividing by K or
using a loose small-gap term. Multiple replacements require their combined
margin error and dependence to be handled explicitly; this proof does not
silently add conditional tails or discard cross terms.

For any task label Y, |P(original correct)-P(surrogate correct)|<=P(D).
There is no independence requirement between Y and either prediction. This is
an interpretation of a DISAGREEMENT bound if obtained, not a computed accuracy
bound for any current model. If the unknown gap term is large, the bound can
be completely uninformative despite a small residual moment.

## 6. Positive-diagonal gauge and scope

Under h'=D h, Wi'=D Wi, Wh'=D Wh D^-1, Wo'=Wo D^-1 with positive diagonal D,
T233 gives u'=u, v'=v, r'=D_j*r and K'=K/D_j. Hence mu'_2=D_j^2*mu_2 and
K'^2*mu'_2=K^2*mu_2. Choosing eta'=D_j*eta preserves the exact tail event and
K'*eta'=K*eta. The surrogate logits, G and disagreement event are unchanged
under the corresponding single-coordinate replacement. The bound is therefore
invariant under this reparameterization; it is not a fixed-weight symmetry.

All statements concern exact real arithmetic and the explicit fresh-input law.
No Gaussian numerical evaluation, state/margin integration, target probability,
implementation cost, estimator gain or universal family exclusion follows.
Missing directions include same/zero slopes, rank-truncated input laws, several
anchors, stable finite arithmetic and a cheap certified gap distribution.
