# Shared-face inclusion and rank-coupled remainder

This theorem is conditional on the pinned, independently reviewed exhaustive correctness-face construction, including its permanent-tie treatment and null input hyperplane exceptions. The present phase checks the new common inequality for every feasible occurrence, rather than replacing that proof with finite witnesses.

Let x0<0 produce hidden state(s,0), s=|Wi0||x0|>0. Write subsequent inputs as(x,u,y), withu>0 and target index2 second-largest. The rank domain is the union{x<u<y} and{y<u<x}; the negative prefix cannot exceedu. Divide input/state coordinates byu and apply the existing target1face certificates.

Every feasible moving correctness face in that complete certificate has first intermediate mask(OFF,ON), ranky<u<x and original inequality Wh10*s+Wi1*x>=0. SinceWi1>0 andWh10<0, x>=kappa*s, kappa=−Wh10/Wi1. For fixed inputs outside{x>=kappa*s,y<u<x}, the same exclusion holds for every s'>=s becausekappa>0. No moving correctness face can be crossed on this connected amplitude ray. The inherited tie argument makes correctness constant along it, apart from the specified Gaussian-null boundary inputs. BothWh column0 entries are negative, so for fixedfinitex all first subsequent preactivations are negative once amplitude is large enough. The limiting intermediate state is therefore exactlyzero, and the constant value equals the reset-limit correctness.

Thus the original and reset-limit correctness indicators can differ only in
E={x0<0,x>k|x0|,0<u<x,y<u}, k=kappa|Wi0|>0.
This includes possible irrelevant faces and need not equal the actual disagreement event.

For independent standard Gaussians, setx=v>0 and integrate the negative-prefix magnitude first, obtainingPhi(v/k)−1/2. Integratingu andy next gives
integral_0^v phi(u)Phi(u)du=(Phi(v)^2−1/4)/2.
Consequently P(E)=(1/2)integral_0^infinity phi(v)[Phi(v/k)−1/2][Phi(v)^2−1/4]dv. There is no normalization by a conditional rank-event probability. The full negative-prefix/positive-target sector has mass1/12, with1/24on each rank side.

Forw>=0, exp(−t²/2)<=1 and>=1−t²/2 imply
phi(0)(w−w³/6)<=Phi(w)−1/2<=phi(0)w.
Integration by parts gives
I1=integral_0^infinity v phi(v)(Phi(v)^2−1/4)dv
=2integral_0^infinity phi(v)^2 Phi(v)dv
=1/(4sqrt(pi))+atan(1/sqrt(2))/(2pi sqrt(pi)).
The last identity follows by integrating exp(−v²)erf(v/sqrt(2)), or the corresponding planar Gaussian angular probability. Therefore the upper linear term is
L=1/(8sqrt(2)pi k)+atan(1/sqrt(2))/(4sqrt(2)pi²k).
For the lower bound, Phi(v)^2−1/4<=3/4 and integral_0^infinity v³phi(v)dv=2phi(0) bound the cubic subtraction by1/(16pi k³). No small-v/k assumption is used. Hence max(0,L−1/(16pi k³))<=P(E)<=L. Independently, the inner rank integral is at most3/8 and the planar wedge probability isatan(1/k)/(2pi), giving P(E)<=3atan(1/k)/(16pi).

At k=1 substitute t=Phi(v): P(E)=(1/2)integral_(1/2)^1(t−1/2)(t²−1/4)dt=11/384, an exact control independent of numerical quadrature.

The exact-rational implementation bounds all constants/functions: Machin's identity forpi; alternating arctan with next-term remainder after monotone argument reduction; dyadic square-root endpoints checked by exact squaring. Rational interval arithmetic then gives outward L and remainder bounds. Numerical adaptive integrals are only cross-checks. A lower bound onP(E) is not a lower bound on actual disagreement.

If a separate phase computes reset-limit joint correctness massJlimit on this sector, originalJ satisfies|J−Jlimit|<=P(E), with both in[0,1/12]. No contribution outside this sector, no fullmodelaccuracy and no production-cost advantage follows without further work.
