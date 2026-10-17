# Independent hidden-reference and deterministic-history review

## T37 residual identity and support rescue

Let hbar_t=B r_t, z_t=W hbar_t+w x_t, D_t=diag(1[z_t>0]). The linearized
surrogate has hlin_next=relu(z)+D W(hlin-hbar). Its deviation from the reference
e=hlin-hbar obeys an independently implementable recurrence

    e_next = [relu(z)-hbar_next] + D W e.

For q=h-hlin and d=W(h-hbar), exact true-minus-linear error is
q_next=D Wq+kappa, kappa=relu(z+d)-relu(z)-D d. Convexity gives kappa>=0,
zero away from crossings and bounded by |d|1[|z|<=|d|]. Signed W means the
accumulated q need not be nonnegative. The reference gate convention at z0
must remain explicit.

Support clipping has an especially simple alternative implementation:

    relu(relu(z)+D W(hclip-hbar)) = D relu(W hclip+w x).

It is the original nonlinear update restricted to reference-active neurons;
reference-inactive activations remain forbidden. This is a genuine plausible
rescue, not an arbitrary output correction. A final exact ReLU step gives
componentwise error<=|W||q_before|, without guaranteeing better labels.

check_rescues_review.py independently implements the residual recurrence and
masked-ReLU support repair on the first512 heldout sequences of each frozen
seed370202/370203, for both noisy models, using saved calibration-only B.
All predictions match the author's implementation; maximum logit difference
8.047e-13. Actual trained forward outputs match exactly. No fit or new input
stream was introduced; these are2048 replayed model-sequences, not additional
independent statistical evidence. Raw comparisons are retained in JSON.

The calibrated B and sampled diagnostic are explicitly not deductive inference.
Full heldout linear and final-exact-step failures remain scoped to that fit
and reference. The fresh support rescue substantially improves both models:
the .0025 case still fails its .001 diagnostic ceiling, whereas the .02 case
is inconclusive at that ceiling. It is incorrect to report all hidden-reference
approaches as rejected. Richer gate/state conditioning and improved alignments
remain untested; signed recurrence and label margins prevent monotone claims.

## T76 deterministic nonzero history and exact two-step mean

For m omitted iid inputs and retained suffix(y,x), write q=Phi. If prediction
is a deterministic function of this suffix and fixed weight-derived history,
its omitted-rank integration remains exact FOR THAT SURROGATE:

- final-input label: x<y integrand q(x)^m, primitive q(x)^(m+1)/(m+1);
  x>y integrand m(1-q(x))q(x)^(m-1), primitive
  q(x)^m-m*q(x)^(m+1)/(m+1);
- first-suffix label: x>y constant q(y)^m; x<y constant
  m(1-q(y))q(y)^(m-1), multiplied by the scalar Gaussian interval mass;
- each omitted label receives one m-th of the remaining rank probability.

Every accepted readout interval must additionally split at x=y. A zero-history
control should reproduce the independently checked two-suffix method. The
real omitted history is correlated with its ranks; replacing it by fixed mu
is the approximation, not an independence theorem about the actual network.

The exact mean after two inputs is available without fitting. For each hidden
coordinate let c+=W relu(w), c-=W relu(-w), and input coefficient w_i. Then

    E h2_i = [hypot(c+_i,w_i)+c+_i+hypot(c-_i,w_i)+c-_i]
             /(2sqrt(2pi)).

Proof: for independent standard-normal X,Y,
E[(cX+wY)+1(X>0)]=(hypot(c,w)+c)/(2sqrt(2pi)); central symmetry makes the
absolute-value term half its unconditional expectation. Sum first-input signs.
For c<0, compute hypot(c,w)+c as w²/(hypot(c,w)-c), with explicit w0 handling,
to avoid cancellation. Tiny T5 input coefficients make this relevant.

Later mean closure sigma*phi(a/sigma)+a*Phi(a/sigma), a=Wmu, sigma=|w|,
is exact only when the previous hidden state is deterministic. It is not the
true recursive mean in general, nor a bound under signed W. Exact T4 prefix
mean therefore removes mean-estimation error but not distributional history
error. No nonzero-history engine is implemented by this review.

## T27 observed-gate phase accounting

For A*=D* W D* and a zero-extended active-block row eigenvector l satisfying
lA*=lambda l, z_t=l h_t has exact decomposition

    z_t=lambda z_(t-1)+l(D_tW-A*)h_(t-1)+lD_t w x_t.

Iteration gives the finite geometric sum from any chosen post-maximum time.
D*W alone is insufficient because inactive columns can act on earlier hidden
states. For real A*, eig(A*.T) supplies the unconjugated complex row convention;
conjugating it conjugates lambda. Observed-gate terms are algebraic accounting,
not causal interventions. Age-conditioned initial phase, final-mask selection
and near-zero amplitudes can shift phase regressions independently of the
identity. No numerical phase experiment was performed here.
