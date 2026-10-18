# mf_verify: T299, independent check (AGENTS rule 5) of the mean-field readout negative (T294/T295)

↑ [Approaches index](../README.md) · [TASKS](../../../TASKS.md) · checked: claude/t194-width16-probe meanfield/readout (T295), meanfield (T294)

## What I read
To match the definitions I read only the **READMEs** of T293 (blind11), T294 (meanfield) and T295 (meanfield/readout)
for the unit definition, the per-unit criterion and the models. I read **none of their code** (blind.py, mf*.py,
readout.py, diag.py). Everything here is my own code.

## Pre-registration (written 2026-10-18, before any T299 number)
DRI brief: "On 50 blind units of 5 DEV2 models: (a) the pass rate of an exact Gaussian orthant on the exact sampled
margin moments (per unit, within 20% relative or .01 absolute); (b) an oracle for 'a Gaussian state at step k is
good enough'. Project the sampled hidden state onto a Gaussian (same mean and covariance) at step 10-m, draw from
it, then run the exact network for the last m steps, for m = 1..4. Report the per-unit pass rate. Pre-register
agreement as pass rates within 10 points of T295's (a: 36%; m=1..3: 18/28/30%)."

**Units** (T293's definition, from its README). Blind region = at most one positive input. A unit is (case, label):
- case q in 0..9 (x_q > 0, the other 9 negative), label j = the position of the largest negative (j != q): 90 units;
- case none (all 10 negative), ordered pair (a = the largest, j = the second largest), label j: 90 units.
The unit's input law is N(0, I_10) conditioned on the unit's event. I sample it exactly: |Z| for the positive and
sorted -|Z| values for the negatives, the top roles placed at their positions and the rest permuted uniformly over
the remaining positions. P(unit) = P(argmax logits = j | unit).

**Models / units.** DEV2 #0, #5, #20, #40, #60 (as T295). 10 units per model, drawn uniformly without replacement
from the 180 with rng 299 (my own draw, not T295's units). **Secondary:** all 180 units of the same 5 models (900),
to separate my pass rate from unit-sampling noise (the binomial SE on 50 units is ~6-7 points).

**Reference P:** 4e5 exact samples per unit (seed A). **Moments / oracle draws:** separate samples (seed B; 4e5 for
the moments, 2e5 oracle draws per (m, variant)).

**(a)** Margins m_i = L_j - L_i (9 of them). Mean and covariance from the seed-B samples. P_hat = P(N(mean, cov) >= 0)
by Genz (scipy multivariate_normal.cdf, abseps 1e-6). Clark is not run.

**(b)** Hidden state h_k after k = 10-m steps (m = 1..4), from the seed-B samples. Gaussian projection with the
exact sampled mean and covariance, then the exact network for the last m steps (inputs x_k..x_9). Two input
treatments, as T295:
- **G:** a joint Gaussian over (h_k, x_k..x_9), with all inputs drawn from it;
- **T:** x_k..x_9 drawn from their true unit law (fresh exact samples), h_k from the Gaussian regression of h_k on
  (x_k..x_9) (exact sampled conditional mean and covariance).
m = 0 (a Gaussian projection of h_10, i.e. Gaussian logits) is reported as a check of (a).

**Criterion** (DRI, as T295): pass iff |P_hat - P| <= 0.2 P, or |P_hat - P| <= 0.01 where P < 0.05.

**Agreement rule:** my (a) pass rate on the 50 units is within 10 points of 36%, and my (b-T) pass rates for
m = 1, 2, 3 are each within 10 points of 18 / 28 / 30%. Note that T295's (ii) used the *mean-field* state, while
mine uses the *exact-moment* Gaussian state (the oracle the DRI asked for), so (b) agreeing means that even the
best Gaussian state does no better. A higher oracle rate would mean T295's (ii) failure is partly the MF closure,
not Gaussianity. (b-G), m = 4 and the 900-unit secondary are reported without a rule.
Verdict wording: agree on all four -> T294/T295 negative reproduced (DRI may mark dead, verified, scoped to Gaussian
states and orthant readouts); otherwise document the discrepancy. Labels: everything here is diagnostic sampling.
Cap 1.5 h.

## Question
Does T295's negative hold under independent code? (a) Is the per-unit margin law given a blind unit too non-Gaussian
for an exact orthant? (b) Is a Gaussian state at step 10-m already too wrong, even with *exact* moments, so that
the exact last m steps cannot repair it?

## Method
`mfv.py` (own code; diagnostic sampling throughout). Exact unit samplers (half-normal positive, sorted negative roles,
uniform permutation of the remaining roles; checked: every draw has the right case, label = 2nd largest, top role =
largest). (a) Genz orthant on the sampled margin mean and covariance. (b) Exact-moment Gaussian projection of
(h_k, x_k..x_9) (variant G) or of h_k given the true late inputs (variant T), then the exact network for m steps.
Shares `../oracle_floor/panel.py` (DEV2 loader). Run: `python mfv.py main` (50 units, 4e5/4e5/2e5 samples), then
`python mfv.py all` (all 180 units x 5 models = 900; smaller samples to fit the cap: 1e5 reference, 1e5 moments,
5e4 oracle draws; this deviates from the pre-registration's sizes, but the secondary has no rule).

## Status
Done (about 1 CPU-h). The pre-registered agreement holds for (a), m=2 and m=3. **m=1 disagrees** (my oracle 36% vs
T295's MF 18%). The discrepancy is in the direction expected from oracle vs mean-field, and it does not change the
negative.

## Results
Pass rate = the fraction of units with |P_hat - P| <= .2P (or <= .01 where P < .05). Median |err| in P in brackets.

| method | **main, 50 units** | T295 (50 units) | agree (+-10)? | secondary, 900 units |
|---|---|---|---|---|
| (a) Genz orthant, exact sampled moments | **44%** (.043) | 36% | yes | 40% (.043) |
| m=0 Gaussian logits by draws (check of a) | 44% | – | – | 40% |
| (b-T) exact-moment Gaussian h_9 + exact last step, m=1 | **36%** (.040) | 18% (MF state) | **no (+18)** | 42% |
| (b-T) m=2 | **38%** (.044) | 28% | yes (edge, +10) | 43% |
| (b-T) m=3 | **40%** (.042) | 30% | yes (edge, +10) | 47% (.040) |
| (b-T) m=4 | 48% (.038) | – | – | 52% (.036) |
| (b-G) m=1..4 (inputs also Gaussian) | 28 / 28 / 22 / 20% | 6-14% (MF) | – | 37 / 32 / 27 / 25% |

By unit type (900): one-positive units a 36%, T1..T4 40/40/44/50%; all-negative units a 45%, T1..T4 44/45/49/53%.
The blind-contribution error per model ((P_hat - P) x 11/1024, weighted as T293, 180 units): (a) -.29 to -.006 pp
over the 5 models, and T4 -.20 to -.005 pp. This is the same sign and range as T295's (-.23 to +.03). The orthant
pulls P toward the middle on the units with large P, which underestimates P(correct).

## Claims
1. [measured, reproduced] The exact Gaussian orthant on exact sampled margin moments fails the per-unit criterion
   on ~60% of blind units (pass 44% on 50, 40% on 900; T295 36%). The margin law given a unit is far from Gaussian.
2. [measured, new] Even an **exact-moment** Gaussian state at step 10-m plus the exact last m steps passes only
   36-48% (50 units) / 42-52% (900) for m = 1..4. That is +2 to +12 points over the plain orthant: about +3 points per
   exact step, as slow as T295 found (~9% error reduction per step). The Gaussian state is already inadequate at
   steps 6-9, even with perfect moments. T295's inferred mechanism is confirmed, with the MF closure removed as a
   confounder.
3. [measured] At m=1 the MF state (T295 18%) is clearly worse than the exact-moment oracle (36-42%). The MF h-closure
   adds error on top of Gaussianity at the last step. At m >= 2 the two are close. This is the documented discrepancy.
4. [measured] Drawing the late inputs from the Gaussian too (variant G) is worse, and gets worse with m (37 -> 25%):
   the inputs' sign-fixed half-normal law matters.

## Alternatives not ruled out
- The units are a different random 10 per model than T295's. The binomial SE of a 50-unit pass rate is ~7 points,
  so the m=1 gap (18 points) is ~2 sigma under pure noise. The 900-unit secondary (42%) supports a real gap, though.
- A Gaussian *mixture* state (split on early gates), skew-normal margin laws, or m >= 5 were not tried (T295's own
  untried list). The oracle at m=4 (52%) is still far from the 90% go line, so m would have to grow a lot (with
  T295's branch counts growing 5-10x per step).
- The secondary uses smaller samples. Its reference SE is .0016 at P = .5, small against the 20% criterion.

## Failure modes
- The per-unit criterion is strict for mid-range P (20% relative). Most failures are units with P in .1-.7, where
  a Gaussian orthant misplaces mass by .04-.1.

## Verdict for the DRI
The T294/T295 negative is **reproduced in substance by independent code**: (a) agrees, m=2/3 agree at the edge,
and the exact-moment oracle (stronger than T295's MF state) also fails at every m <= 4. The pre-registered rule
flags m=1 (36 vs 18), which is explained by the MF closure's own last-step error, not by Gaussianity. Suggested
scope for "dead (verified)": Gaussian-state closures with an orthant readout on the blind units, with up to 4 exact
last steps. The mixture-state route stays open.
