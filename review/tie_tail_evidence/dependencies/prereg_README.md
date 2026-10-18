# tie_tail_labels (T218): is the heavy tail of the tie-slicing integrand a few mechanism-labelled families?

↑ [Approaches index](../README.md) · [INDEX (registry + ledger)](../../../INDEX.md) · [TASKS](../../../TASKS.md) ·
[sci_tao (E25)](../sci_tao/README.md) · [ht_seeds (E33)](../ht_seeds/README.md)

## Question
Deterministic tie-slicing (E25/E33) loses its edge at T~5-8 because the face integrand is heavy-tailed (top 5% of
tie points carry ~35% or more of the second moment, H-E12). Direction D2 asks whether that tail is made of a FEW
mechanism-labelled families. If it is, each family could get a dedicated deterministic rule.

## Pre-registration (written 2026-10-18, before any measurement)

**Integrand.** sci_tao's exact co-area identity, copied unchanged (`core.py`): err = Z23 E23[F] + Z12 E12[F].
With n faces per kind, the per-point contribution is g = Z_kind * F. Tail = the top 5% of all 2n points by g^2.
Tail mass = their share of sum g^2.

**Primary label (L_mech)**, computed per tail point from an exact re-walk of its segment:
(kind, positions (a, b, p1) [these fix the pair distance |a-b| vs clock period, the age of the max, record order and
the x_{T-1} case], value bucket of the 2nd-largest value u (<=0 = blind, 0-0.25, 0.25-0.5, >0.5), wrong-answer
type (tie partner / max / 3rd / other), and the set of (step, neuron) gate flips strictly inside the dominant
misclassified interval (up to 3; "none" = the band sits in one linear cell)).
Coarser schemes are reported too (positions only; positions+flip set; mechanism without positions). At T=6 there
are only 75 position configurations, so any scheme coarser than L_mech passes "<=100 labels" trivially. Those
schemes are descriptive and do not decide.

**Primary metric: out-of-sample coverage.** Choose the top-L labels by tail mass on rule A (Sobol seed 700, 2^14
faces per kind). Then measure the share of rule B's tail mass (seed 701, independent scramble) that falls in those
labels, for L = 10, 30, 100. This guards against the trivial in-sample case where every label is a single point.

**Decision rule** (L_mech, out-of-sample, L = 100):
- **Go**: coverage >= 90% on every model: zoo (16,6) seeds 0-4, target M_16,10, and noisy copies sigma 0.005 and
  0.01 (sigma x matrix RMS, seed 0).
- **Kill**: coverage < 70% on the target or on either noisy copy.
- Otherwise **inconclusive**, with what would separate the cases.

**Controls.**
- Singleton null: top-L individual points (what any too-fine labelling achieves in sample).
- Random-label null: the same label-size profile with labels randomly reassigned.
- Sample-size check: in-sample share at 2^13 vs 2^14 faces. Real families should give a share that is stable in n.

**Alternative explanations to separate (AGENTS rule 3).**
1. *Few families* (the D2 hope): coverage is high out of sample and stable in n, the same labels lead across noisy
   copies, and the integrand within a label is light-tailed.
2. *Continuum of rare events*: the tail is a spread of near-unique (position, value, flip) combinations. Then
   coverage is low out of sample and the in-sample share falls as n grows.
3. *Few coarse families that are still heavy-tailed inside*: labels cover the mass but the within-label tail
   share stays near the global one. Then the labels would not give a usable dedicated rule. Test: the within-label
   top-5% share for the top labels.
4. *Model-specific labels*: concentration holds on each model, but the leading labels differ between orig and its
   noisy copies, so no rule is fixed across copies. Test: overlap of top-100 label sets across copies.

**Blind sub-test.** Exact prefix region counts of the target restricted to {at most 1 positive coordinate}, k = 1,
2, ... Go if the region growth is <= 4x per step.

## Method
(filled in below as stages complete)

## Status
Running.
