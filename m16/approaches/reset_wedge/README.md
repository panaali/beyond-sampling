# T49 magnitude-dependent H2 reset wedge

## Question
Can a weight-derived reset region larger than two negative inputs reduce unresolved history for the trained H2T8 estimator? This lane is distinct from state-density transport, history moment closure and tail integration. Claim e30f4b9 was pushed before implementation. Claude remains DRI.

## Method and frozen algebra
Assume Wi=(w1,w2)>0 and Wh=[[-a,-b],[c,-d]] with a,b,c,d>=0. The initial hidden state ranges over the entire nonnegative quadrant, a stronger domain than the reachable set. Define Gamma=c*w1 when d=0; otherwise Gamma=max(0,c*w1-d*w2,c*max(w1-b*w2/d,0)). Proposed exact uniform-state reset set for a pair(x,y) is y<=-kappa*max(x,0), kappa=Gamma/w2.

Proof candidate: after the first input, maximize c*h1-d*h2 over old state(u,v)>=0. Increasing u cannot increase the objective, so u=0 attains the maximum. For x>0, remaining v dependence is piecewise linear with breakpoints w1*x/b and w2*x/d; its maximum occurs at v=0, v=w2*x/d, or the nonpositive/zero branch. For x<=0, h1=0 and the maximum is0. At the second input the second gate vanishes for every old state exactly when w2*y+Gamma*max(x,0)<=0. Since Gamma>=0 this also makes y<=0, forcing the first gate off. Equality resets exactly; there is no epsilon test.

For iid standard Gaussian inputs, q=atan(kappa)/(2*pi), reset-pair mass r=1/2-q. Let S_n be probability of no reset among n inputs. Proposed exact recurrence S0=S1=1; S_n=.5*S_(n-1)+q*S_(n-2). The surviving positive final-input density is phi(x)*S_(n-1); negative x requires previous positive y>-x/kappa, whose surviving density is phi(y)*S_(n-2). For a last reset ending at j>=2, mass is r*U_(T-j), where U0=1,U1=.5 and the same recurrence holds. These masses plus S_T must sum1. kappa0 and kappa→infinity yield reset-on-any-negative and the original two-negative rule respectively.

## Frozen falsifiers and alternatives
1. The proposed wedge may only suffice, or may omit an optimizer/degenerate gate case. Independently maximize the piecewise-affine objective by enumerating all quadrant/activation-region vertices using rational arithmetic, compare synthetic positive/zero a,b,c,d cases, and construct violating states just outside the candidate boundary. No fitted coefficient.
2. Reset probabilities may be incorrectly multiplied despite overlapping pairs. Independently integrate the one-dimensional survival operator at small n using positive adaptive quadrature, plus exact limiting recurrences and total last-reset partition. No Monte Carlo estimator or trained accuracy sweep.
3. The enlarged wedge may add negligible probability at actual weights. Freeze H2T8 seed0 and the two existing relative-noise copies(.0025,.02; historical RNG1 protocol) before measuring coefficients/masses; preserve failure of sign assumptions. Verify checkpoint/model hashes and exact positive diagonal/input rescaling invariance.
4. Resetting hidden state does not erase input rank. The enlarged event couples the two discarded inputs; old independent-negative prefix rank laws cannot be reused. Keep all unresolved rank/history mass explicit. First phase delivers a reset theorem and mass inventory only, never a full accuracy claim.

## Compute accounting
Each cycle<=110CPU, one thread/2GB; primary compute lock for long runs, review lock only<=20CPU checks. Initial phase target<=20CPU. Rational operations/bit lengths, adaptive scalar callbacks, arithmetic/special-function proxies and measured CPU reported separately. No silent conversion of exact-rational operations into FLOPs. Source/review diagnostics are not production cost. No shared-source edits, no sampling advantage claim before independent review.

## Status and claims
Status: frozen theory/control phase. [inferred] Piecewise linear optimization and recurrence may give exact real-arithmetic reset masses; proof awaiting independent review. [speculative] The additional reset mass may materially shorten suffixes. Numerical integration, floating probability evaluation and trained usefulness are not yet measured.

## Alternatives not ruled out and failure modes
Reachable-state restrictions could yield an even larger reset set. Unsupported weight signs need a different theorem. Small input weights can make kappa huge and gains negligible. Rank dependence, long surviving suffixes and conditional integration error may dominate even if resets improve. Exact algebra does not certify floating numerical evaluation; no method family is declared solved or dead here.
