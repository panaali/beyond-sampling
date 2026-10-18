# Exact network-sector assembly gate

Scope d7c0edab. Six inherited analytic H4 fixtures only, no trained checkpoint.
For x1=sign*u with u>=0, h1=relu(sign Wi)u. Exact stored-weight arithmetic gives
a=Wh relu(sign Wi), then h2=relu(a u+Wi x2). For every nonzero second coefficient
b, the interior gate ray has exact slope -a/b. Sort these rational slopes and
merge equal rays; add(0,-1),(0,1). A positive sum of adjacent rays gives a strict
interior point, except the whole-halfplane case uses(1,0). Gate activity is an
exact rational sign, including structurally zero rows. Two sector normals are
(-lo_y,lo_x,0),(hi_y,-hi_x,0). h3 unit coefficient is(Wh A row,Wi unit).

The two input1 signs contribute their actual Gaussian half-sector measures;
there is no additional factor1/2. Boundary ownership is immaterial to means
but deterministic exact boundary/interior direct forwards must match both h2
and h3 maps. Exact comparisons/sorting/quotients and dot-product counts are
qualified separately; this is not free model advice or a floating sector proof.

Controls are the six frozen prior T69/recovery fixtures: zero, zero recurrent,
identity, negative identity, duplicate/opposite and mixed rational weights.
Every sector/unit gets fixed160/200-digit facet evaluation and independent
80-digit radial-angle reference, with absolute1e-60 plus relative1e-50 target.
Zero/zero-recurrent means have immediate analytic values; identity recurrence
has mean |Wi|*(1+1/sqrt2+1/sqrt3)/sqrt(2pi) by the reviewed three-step Spitzer
identity. Precision stability and analytic equalities use1e-100*(1+abs(target)).
These are diagnostic tolerances, not outward bounds.

Append attempts and cell results; preserve all partial rows/failures and refuse
existing outputs. One thread2GB, review<=20CPU; if capped, preserve unresolved
work rather than increasing precision or rerunning completed cells. Independent
source/control review required before execution and saved audit before any
trained mean. Historical T69 mean/covariance/closure errors remain unchanged.
