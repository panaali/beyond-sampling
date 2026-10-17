# Conditional segment templates

Let h[j]=D[j](W h[j-1]+w x[j]), with D[j] a fixed diagonal0/1 gate matrix.
Take local coordinates z=(h_in,x[1],...,x[k]). Set A[0]=[I,0]. Recursively,
B[j]=W A[j-1]+w e[H+j-1]^T and A[j]=D[j]B[j]. Then h[j]=A[j]z on the
specified gate region. This follows by substitution, including arbitrary
incoming states; no assumption h_in=0 is made.

For every neuron, the region requires B[j,i]z>0 when ON and <=0 when OFF.
An identically-zero form requested ON makes that word infeasible. An identical
zero OFF form is a full-dimensional equality, not a measure-zero boundary.
Nonzero forms vanishing at a witness are retained as boundaries. The templates
are meaningful algebra even for infeasible words, but those words do not count
as actual network regions or reusable visits.

For absolute start s, insert local inputs into global positions s..s+k-1 by
a linear coordinate-selection matrix E[s], leaving incoming hidden coordinates
unchanged. All maps/planes become A[j]E[s],B[j]E[s]. Thus their local rational
coefficients are independent of s for fixed W,w and gate word. Readout rows and
target ranks are not automatically shifted; they are separate data.

For adjacent words u,v, substitute the output map of u into the incoming-hidden
columns of v and place v's local inputs after u's inputs. The output equals the
direct uv template. Its region is the conjunction of u's planes and the
substituted v planes, with identical strict/weak ownership. This supplies a
potential segment cache, not a universal common cone or subtree.

Two obstructions must remain explicit. With scalar W=w=1 and an ON step,
the formula h+x is shared, but its input domain x>-h differs with incoming
h. Taking nonnegative incoming states1 and2 at input-3/2 gives different
actual gates. Incoming states of real recurrent prefixes must be nonnegative;
the general affine identity also permits arbitrary algebra-control coordinates. With scalar W=0,w=1 and fixed last positive input1, sequences(3,2,1) and
(2,3,1) have the same final hidden state and gate history, yet different
second-argmax truth labels. A fixed output index0 is wrong on the first and
correct on the second. Neither map reuse nor terminal-state equality alone
licenses reuse of correctness probability.

Any trained reuse census must distinguish unique histories from copies across
initial rank charts, conditional neural maps from cone/rank domains, and
template arithmetic from geometry/mass cost. Cache construction/key lookup,
exact bit widths, substitutions, remaining horizon and readout/target handling
remain part of any later complete cost comparison. No exhaustive impossibility
or universal speedup follows from this initial theorem/control phase.
