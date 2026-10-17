# Named arithmetic for a prefix-shared segment cache

This is a cost-first, saved-history discriminator, not a production cache.
A history is a sequence of H-bit gate masks. P_t is the set of distinct
prefixes of length t, including the empty prefix. Current gate signs do not
change current preactivation B_t, so B_t is computed once per P_t, NOT once
per child mask or complete history. Initial rank charts and duplicate
simplices do not supply additional histories.

Use h_0=0, and retain only the t existing input columns before step t.
The new Wi column is copied; future columns are identically zero. A dense
H-by-H times H-by-t product uses H^2*t multiplications and H*(H-1)*t
additions. Define c=H*(2H-1). The baseline has named arithmetic

    c * sum(t*|P_t|, t=0..T-1).

Masks and output pointers require copying/control, separately unpriced.
This baseline already exploits zero initial state, triangular input support,
and common prefixes, but not coefficient sparsity, reset-state equality,
low-rank multiplication, or specialized H=1 algebra.

For EACH fixed cut a in1..T-1, let b=T-a. Cache a symbolic suffix
preactivation for every distinct suffix parent-prefix S_j={word[a:a+j]},
j=0..b-1. Its incoming-state coefficients have H columns; its earlier local
inputs have j columns. First preactivation (j=0) simply copies Wh and Wi.
At j>=1 building a cached preactivation costs c*(H+j) per S_j. The cached
state needed by the next prefix is obtained by applying the mask; current
Wi again appends by copying. No matrix is built per complete suffix word.
Thus cold cache arithmetic is

    c * sum((H+j)*|S_j|, j=1..b-1).

At global depth t=a+j, its cached preactivation's H incoming columns multiply
the prefix-state map A_a (H-by-a). The j earlier suffix columns and new Wi
column are appended unchanged. This costs c*a per GLOBAL parent-prefix P_t.
Both the state and its gate planes must be transported; the final masked
state aliases the last plane instead of requiring a duplicate transform.
The same substitution for a shared global parent-prefix is performed once.
Warm arithmetic, including direct prefix construction, is

    c * (sum(t*|P_t|,t<a) + a*sum(|P_t|,a<=t<T)).

Report warm, cold, and their sum for EVERY frozen cut. Warm-only savings do
not establish cold savings. A virtual scalar loop over the specified dot
products independently contracts the source-shape counts on controls and
saved histories; it does not execute trained matrices or simulate native
operations. Numbers are projected named multiplications/additions, not
actual runtime or evaluator cost. Histories are advice; discovering them,
canonical keys, sparse/reset handling, copies, storage, Fraction GCD and bit
complexity, validation, cone intersections, truth/readout and probability
integration remain explicit obligations, not free operations.

An improvement in this limited arithmetic permits a separately frozen
implementation experiment. A loss does not exclude other cuts at other
sizes, multiple segment boundaries, shared-substitution factorizations,
sparsity/reset reuse, broader affine equivalence or domain-restricted reuse.
