# skeleton_quadrature (T103): deterministic quadrature of the top-3 skeleton integral

↑ [Approaches index](../README.md) · [INDEX](../../../INDEX.md) · [TASKS](../../../TASKS.md) · predecessor: [review/stoch_margin (R5)](../../../review/stoch_margin/README.md)

## Question
Write p = E_S[q(S)]. The skeleton S is made of the top-3 positions (720 ordered triples) and their values
v1 > v2 > v3. q(S) = P(correct | S) integrates out the 7 low inputs.

Can a deterministic rule over S reach a >20x MSE gain over Monte Carlo on this integral (at equal node count) on DEV
models? It may share structure across the 720 position triples: symmetries, shared position classes, or slicing in
the tie direction (v2 - v3). If it can, does that turn into a win over plain sampling at 16k-64k FPE, with the
cost of q(S) counted honestly?

## Pre-registered alternatives and cheapest tests (AGENTS rule 3; written before building)
1. **q is smooth in the values once the low inputs are integrated.** The 2-3 tie point d* moves with the low
   inputs, so q(d23) is the CDF of d*, with a width set by the low inputs. A few Gauss nodes per triple would then
   suffice. **Against:** the rare wide bands (H-E12) live in the low inputs, so 1-q(d23) could have a long, heavy
   tail. *Test D1:* profiles of q along d23 and along the other value coordinates, with M = 2048 inner draws; and the
   share of the error mass beyond each d23.
2. **Positions enter through a few features.** For example only (p2, p3), only the time order, or only the
   distance to the end. Then one value-rule can be shared by many triples. **Against:** each triple is its own
   function. *Test D2:* q(pos, v) for all 720 triples at fixed value points; the between-triple variance explained
   by each position feature set.
3. **Even a perfect skeleton rule cannot win much end to end.** The inner integral (7 low inputs per node) keeps
   its own error. If it is done by MC/QMC with n points per node at total budget B, its variance is at least
   E_S[q(1-q)] / (g_in B), where g_in is the inner rule's gain over iid. So the end-to-end gain is at most
   g_in / (1 - leverage). From R5, leverage is 0.65-0.73, which gives 2.9-3.7x x g_in. *Test D0:* measure g_in
   (Sobol vs iid inner points, per skeleton).
4. **The skeleton rule wins only through stratification of the 720 triples.** That would be at most
   1/(1 - between-share). *Test:* compare a stratified-MC rule (each triple visited, random values) with the
   deterministic value rule.

## Status
Running (2026-10-17). Diagnostics first.
