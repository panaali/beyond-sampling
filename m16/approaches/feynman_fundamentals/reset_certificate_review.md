# Exact reset certificates: two-input wedge and projective word

T55 algebra-only reviews supporting separately owned T49 and T50, 2026-10-17. No reset solver or probability engine implemented here.

## Positive-input-weight inhibitory wedge

Let Wi=(w1,w2)>0 and Wh=[[-a,-b],[c,-d]], with a,b,c,d nonnegative. After input x from old state(s,t)>=0, the second preactivation before the next input contains

`F=c*ReLU(w1*x-a*s-b*t)-d*ReLU(w2*x+c*s-d*t)`.

F is nonincreasing in s, so its supremum occurs at s=0. If x<=0 the supremum is zero. If x>0, the scalar piecewise-linear function in t has only two potentially positive vertices: t=0 and t=w2*x/d when d>0. Thus its supremum is Gamma*x, with Gamma=max(0,c*w1-d*w2,c*max(w1-b*w2/d,0)); when d=0 it is c*w1*x. Input y<=-Gamma*max(x,0)/w2 makes both next preactivations nonpositive for every old state.

For independent standard Gaussian inputs, the certified pair event has probability r=1/2-q, q=atan(Gamma/w2)/(2*pi). Avoidance probabilities satisfy S0=S1=1 and S_n=S_(n-1)/2+q*S_(n-2). Positive last input always avoids this pair event. A negative last input survives only with a positive predecessor greater than -x/kappa, giving unnormalized density S_(n-2)*phi(x)*(1-Phi(-x/kappa)); for x>0 it is S_(n-1)*phi(x). Treat kappa=0 separately. After a certified pair, the last input is negative; future avoidance has U0=1,U1=1/2 and the same recurrence. Therefore last certified pair ending j has mass r*U_(T-j).

These are probabilities of the specified two-input certificate. They are not probabilities of never reaching zero hidden state: from h0=0 a negative first input already gives zero, whereas S1=1. Rank/prefix coupling is not eliminated by this reset identity.

## Fixed-word global reset

Represent arbitrary finite old hidden state by y=(s,h01,h02)>=0 with sum one and s>0; multiply every fixed word input by s. Positive homogeneity makes the propagated hidden map continuous and piecewise linear on this compact triangle. Points with s>0 are dense. If exact gate cells cover the triangle, the final map vanishes for every finite old state iff it vanishes at every vertex of every full-dimensional cell. Retain recession vertices s=0. Discarding zero-area cells is justified only through continuity and coverage by closures of full-dimensional cells.

For robustness, on each last-step preactivation piece require L_i(v)<=-eta_i*s(v) at all vertices. Finite vertices give eta_i=min(-L_i(v)/s(v)); recession vertices require L_i(v)<=0. Convexity extends this inequality over each cell. An input-word sup-norm perturbation epsilon changes last preactivation by at most s*epsilon*sum_r |Wh|^r|Wi| componentwise, independently of old hidden state. Choosing epsilon below all positive eta_i/C_i therefore certifies an open reset box without assuming gates stay fixed. A failed strict margin is only failure of this sufficient robustness certificate; it is not proof that no robust reset exists.
