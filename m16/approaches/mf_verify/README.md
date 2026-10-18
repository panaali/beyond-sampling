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
