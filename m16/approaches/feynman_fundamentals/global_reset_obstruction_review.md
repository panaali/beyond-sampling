# T50: minimum positive-probability global zero-reset length

This independently reviews `global_reset_words/obstruction.py` at bcd3311
and the exact projective implementation at f1391f2. It is a theorem about
uniform resetting to the zero hidden state, not forgetting to an arbitrary
common nonzero state, output-label forgetting, or erasure of rank memory.

Write W=[[a,-b],[c,-d]], w=(-alpha,beta), with alpha,beta,b,c,d positive,
a>d and D=bc-ad>0. Fix any two real inputs x1,x2 and any desired r>=0.
Set t=r-beta*x2 and choose v sufficiently large, with
u=(t+d*v)/c. The first hidden state (u,v) is positive. The second
preactivation has second coordinate r and first coordinate
`a*t/c-alpha*x2-(b-ad/c)*v`, which is negative for sufficiently large v.
Thus the second state is (0,r).

The first-state preimage is exactly

```
h01 = [-d*(u+alpha*x1)+b*(v-beta*x1)]/D
h02 = [-c*(u+alpha*x1)+a*(v-beta*x1)]/D.
```

Its coefficients of v are `(b-d^2/c)/D` and `(a-d)/D`, both positive:
bc>ad>d^2. Hence it is a finite nonnegative initial state for large v.
The source's five thresholds, followed by adding one, enforce all these
inequalities with exact Fraction arithmetic and have the correct signs.
Every fixed pair therefore has an image containing the entire second axis.

For any third input x3, choose
`r>max(0,-alpha*x3/b,beta*x3/d)` to force state three to zero. Every
fixed four-word with x4!=0 consequently has a finite initial-state
witness whose fourth state is ReLU(w*x4), which is nonzero. Its word
cannot be a uniform zero reset. The set of potentially resetting
four-words lies inside x4=0 and has Gaussian probability zero.

For completeness, two-word images cannot be singleton zero because they
contain the entire axis. For three words with x3!=0, choose the reachable
axis origin after two steps; its next state is nonzero. One step likewise
has a nonzero recession image. Thus no length at most four has a
positive-probability global zero-reset event. Combined with the separately
certified open five-word reset box for the stored T10 weights, the minimum
positive-probability length is five. Applicability requires the exact
stored-weight sign inequalities, not rounded empirical sign evidence.

The projective solver retains s=0 recession vertices and checks linear
maps on every full-dimensional exact cell. A nonzero recession vertex
plus a finite vertex produces a finite witness; nonnegative output maps
prevent cancellation in this sum. Checking last-preactivation finite
vertex margins and nonpositive recession margins supports the stated
input-box Lipschitz certificate. Boundary cells follow by continuity.
The constructor's missing row-count defense was reported and repaired.

The owner's 74 exact witnesses include fresh rational parameters and
extreme/mixed words; this review checks the construction algebra and
source independently, without rerunning that same control script. The
universal construction, rather than a finite search over words, supplies
the lower bound. Nonzero coalescence remains an open, different question.
