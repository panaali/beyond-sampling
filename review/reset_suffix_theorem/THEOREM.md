# T207: generic equality of forced gate-word state maps

## Scope and notation
Fix integers H>=1 and L>=1. A word D=(D_1,...,D_L) consists of diagonal
H-by-H matrices with entries in {0,1}. All words have the same depth and the
same ordered scalar input coordinates x_1,...,x_L. With a shared recurrent
matrix W and input column v, define the **forced** recurrence

    h_0=0,  h_t=D_t(W h_{t-1}+v x_t),  h_L=A_D(W,v)x.

This is the linear extension associated with a gate word. We make no assertion
that its word is feasible for any ReLU input, or that any observed cone is a
complete representation of that feasible set. W is tied across time. Entries
of A_D are polynomials in the H^2+H independent coordinate variables (W,v);
these variables are coordinates, not a claim about statistical independence
until the probability corollary below.

For two different words D,E, let j=max{t:D_t!=E_t}. Later masks agree. Say the
pair is **reset equivalent** when some t with j<t<=L has D_t=E_t=0 (the entire
hidden mask is off). Identical words are also reset equivalent by definition.

## Theorem
There exists a proper algebraic exceptional set, contained in a finite union
of zero sets of nonzero real polynomials, such that outside it, simultaneously
for every pair of words of length L,

    A_D(W,v)=A_E(W,v)  iff  D and E are reset equivalent.

The forward reset implication holds for EVERY W,v. The converse is a generic
statement only. It concerns equal global linear maps, not agreement restricted
to a word's domain, equal predictions, approximate equality or cost savings.

## Proof: resets suffice
For identical words the recurrences are identical. Otherwise choose a common
all-off mask t>j. Then h_t=0 under both words for every input, regardless of
their earlier states. Since all masks after j, hence after t, agree, the later
recurrences and input injections agree. Their final maps are identical for
all W,v. Several resets or a reset at the last step cause no exception.

## Proof: absent a later reset, a polynomial separates the maps
The coefficient of fresh input x_j in their difference is

    b(W,v)=D_L W D_(L-1) W ... D_(j+1) W (D_j-E_j)v,

with empty left product at j=L. No earlier state or mask contributes to this
column: h_(j-1) depends only on x_1,...,x_(j-1). All factors after j are common.
Choose k where the two diagonal entries at j differ, so
(D_j-E_j)e_k=s e_k with s in {+1,-1}. Evaluate at v=e_k.

If j=L, b=s e_k, which is nonzero.
If j<L, evaluate W at the all-ones matrix J. Let d_t=D_t 1 and n_t=1^T d_t.
Since no later common mask is all-off, n_t>=1 for every t>j. Repeated use of
Jz=(1^T z)1 gives

    b(J,e_k)=s [product over t=j+1,...,L-1 of n_t] d_L !=0.

The empty product when L=j+1 is1. Choose an active final coordinate r. The
r-th component b_r is therefore a nonzero polynomial in the entries of W,v.
Any global map equality for this pair implies b_r=0. This constructs one
proper polynomial zero set containing the pair's accidental equalities.
There are finitely many words (2^(H L)) and pairs, so the union of these
exceptional sets has Lebesgue measure zero. Off this union, no non-reset pair
has equal global maps, proving the simultaneous statement.

The witness uses ONE matrix J at every time: it does not untie recurrence.
J may be singular and v may contain zeros; that is harmless for proving a
polynomial nonzero. No witness feasibility is required. If restricting to an
open set of invertible matrices and nonzero input entries, a nonzero polynomial
still cannot vanish on that open set; a small perturbation of the witness also
retains a nonzero selected component. This is not a trained-model construction.

## Probability corollary and its limits
If the joint law of (W,v) is absolutely continuous with respect to Lebesgue
measure on R^(H^2+H), the theorem's simultaneous conclusion holds with probability
one. Independent entries with absolutely continuous densities, including fixed
means plus independent Gaussian perturbations of positive variance, suffice.
Independence is sufficient, not necessary: joint absolute continuity is enough.
Merely continuous marginal CDFs are NOT the stated hypothesis. Tied temporal
reuse of W is allowed; structural parameter tying or lower-dimensional weight
laws need not satisfy the hypothesis.

This does not assert the property for a particular trained checkpoint, rounded
binary weights, discrete or zero-variance noise, or an arbitrary law supported
on singular/structured matrices. Their finite collisions must be checked, as
T188 did. Finite observations alone do not prove this theorem or its probability
hypothesis. The Gaussian-noise corollary refers to ideal real weights, not the
implemented finite-precision perturbation pipeline.

For example, outside the generic regime let H=2, W=0, v=(1,-1), L=2. The words
D=((1,0),(1,0)) and E=((0,1),(1,0)) have no later all-off mask, yet both forced
maps are h_2=(x_2,0). Both words have actual positive-volume ReLU domains
(x_1>0,x_2>0) and (x_1<0,x_2>0), respectively. This is a structured-weight
exception, not a counterexample to the theorem.

## Connection to the previously reviewed one-step margin theorem
The T188/T11597 sufficient result requires invertible W, every v_i nonzero,
and injective subset sums of the vector hinge fingerprints |v_i| times each
readout-margin column. Under those separately verified conditions, equality
of global one-step future margin functions, or equality up to ONE common
positive constant, is equivalent to equality of their incoming state maps.
Combining that implication with THIS theorem is legitimate only when BOTH
sets of assumptions hold. We do not establish the fingerprint conditions from
this theorem or infer them for other checkpoints/horizons. No claim follows
for equality of argmax labels, input-dependent rescaling, domain restrictions,
convex union feasibility, geometric merging cost or total accuracy.

## Edge cases and alternatives
H=1: every nonzero mask is1, so the same witness and reset argument apply.
L=1: two different masks have nonzero-polynomial difference (D_1-E_1)v; there
is no later reset. L=0 has only the empty word and is trivial, outside the stated
positive-depth theorem. A common final zero mask erases every earlier difference.
An earlier zero mask at/before j does not by itself erase the fresh x_j column.

The two alternatives this proof separates are (i) polynomial cancellation
introduced by tied recurrence, ruled out generically by the tied J witness,
and (ii) actual structured-weight cancellation, retained explicitly and
illustrated above. Different formulas can still agree on restricted domains,
produce the same labels, or be close enough for a certified approximation.
None of those alternatives is excluded; no method family is declared dead.
