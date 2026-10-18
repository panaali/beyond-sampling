# A reachable counterexample to one specific summary

Written derivation only, under claim/delegation7bce65ef. No code, numerical evaluation or change to the completed symbolic suite. All quantities below are exact rationals or formal Gaussian CDF values. Indices are zero-based; argmax and order-statistic ties use first-index ownership.

## Fixed network and two prefixes

Take H=4,T=4, zero initial hidden state, Wh=I_4, Wi=(1,-1,0,0)^T and readout rows

- Wo[0]=(1,0,0,0),
- Wo[1]=(0,1,0,0),
- Wo[2]=(3/4,3/4,0,0),
- Wo[3]=(-1,-1,0,0).

Fix p in {3/4,5/4} and prefix (X0,X1,X2)=(2,p-2,0). The first hidden state is (2,0,0,0), the second is (p,2-p,0,0), and input0 at the third step leaves it unchanged. Both first two hidden coordinates are strictly positive. These are actual zero-start reachable prefixes of this same synthetic network, not arbitrary extended-kernel states. The two prefixes are (2,-5/4,0) and (2,-3/4,0); in each the strict top-two state is a=2 at index0 and b=0 at index2.

The summary from the reviewed primary plan is this exact rank state together with current target margin

m_b(h)=Wo[2]h-max_{j!=2}Wo[j]h.

The current logits are (p,2-p,3/2,-2). For both p values the greatest rival logit is5/4, so m_b=1/4. Thus the proposed summary is exactly identical for the two reachable prefixes.

## Exact last-input decisions

Let the final independent standard Gaussian input be y and write s=p+y. The last hidden state is (max(s,0),max(2-s,0),0,0). For 0<=s<=2 its logits are (s,2-s,3/2,-2). For s<0 only the second hidden coordinate is positive, so label1 beats its3/4 multiple and the zero/negative competitors. For s>2 only the first hidden coordinate is positive, so label0 wins. Combining these cases gives the global prediction:

- label1 when s<=1/2;
- label2 when 1/2<s<3/2;
- label0 when s>=3/2.

At s=1/2 labels1 and2 tie and index1 wins; at s=3/2 labels0 and2 tie and index0 wins. The fourth logit never wins. These are exact statements for all real y, not a sampled check.

For the final second-largest index, the strict open rank regions are label2 for y<0, label3 for0<y<2, and label0 for y>2. At y=0, the old index2 owns second place over the new index3; at y=2, old index0 owns first and the new index3 is second. Hence the exact correctness event for either stated p is

(1/2-p,0] union (2,infinity).

The endpoint0 has zero Gaussian mass; the lower boundary is incorrect because its predicted label is1. The input2 boundary is incorrect because its true label is3 but prediction0. No permanent tie is discarded.

Consequently the conditional success probability is

q(p)=Phi(0)-Phi(1/2-p)+1-Phi(2)
    =1/2+Phi(p-1/2)-Phi(2).

Thus q(5/4)-q(3/4)=Phi(3/4)-Phi(1/4)>0, since the standard Gaussian density is strictly positive on (1/4,3/4). No numerical CDF approximation or value is needed.

## What this disproves and what it leaves open

[inferred] No single function of this exact top-two/current-target-margin summary can equal q at BOTH stated reachable states of this particular network. Any value assigned to their common summary has maximum error on these two points at least [Phi(3/4)-Phi(1/4)]/2, by the triangle inequality. This is a two-point minimax statement only.

The exact specified Gaussian prefixes have probability zero. We do NOT infer a positive-measure lower bound, an accuracy bias, an MSE lower bound or failure of almost-sure sufficiency from these two points alone. Conditional densities on summary fibres and neighbourhood control would require a separate proof. The network is synthetic and is not a trained checkpoint or noisy-copy result. No timing, compression rate or construction-cost improvement was measured.

Richer summaries can distinguish the states; for example the additional value h0 differs. Controlled approximations, signed integration of errors, alternative rank summaries and weak success representations remain open. This refutes only universal pointwise EXACT sufficiency of the particular proposed summary, not the weak-success approach or any approximation/compression family. The prior possible-injectivity caveat remains relevant on other reachable branches and networks.
