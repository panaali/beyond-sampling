# smooth_verdict: T319, Fable r4 D3: Gaussian-smoothed closure verdict (oracle floor test)

↑ [Approaches index](../README.md) · [TASKS](../../../TASKS.md) · design: claude/project-handoff-priorities-zpzfm8 review/fable_review_r4/README.md D3

## Pre-registration (written 2026-10-18, before any T319 number)
**Question.** A point closure pins the lows at their conditional mean mu and outputs the hard verdict 1[m > 0]. Does
replacing it with Phi(m / (s ||J||)) cut the oracle floor? Here s^2 is the lows' exact conditional variance given the
skeleton, and J is the margin's gradient with respect to the low positions at the closure point, from the exact
gate-pattern map (an adjoint pass).

**Closures** (skeleton exact, lows pinned):
- **R0:** top-3 exact; the 7 lows at mu(v3) = E[X | X < v3], s^2(v3) = Var(X | X < v3) = 1 - v3 lam - lam^2, lam =
  phi(v3)/Phi(v3).
- **top-4 (T298 b):** top-4 exact; the 6 lows at mu(v4), s^2(v4).

**Verdicts at the closure point x_c.** L = logits(x_c), label j = position of the 2nd largest, rival i* = argmax_{i != j} L_i,
m = L_j - L_{i*}, J = (A_j - A_{i*}) restricted to the low positions, A = the exact local logit map of x_c's gate pattern.
- hard: 1[m > 0] (the closure's own verdict, baseline);
- **smooth (primary, Fable's formula):** Phi(m / (s ||J||));
- smooth-all (secondary, labelled): prod over the 9 rivals i of Phi(m_i / (s ||J_i||)) (independence across rivals).

**Oracle protocol (diagnostic sampling, paired, as T292/T298).** Per DEV2 model, 250k inputs x ~ N(0, I_10) (seed
319000 + index), ok(x) = the model is correct on x. Per-model floor b_m = mean(verdict(closure of x) - ok(x)), in pp.
All 80 DEV2 models, 1e8 truths not needed (paired). Report RMS_m, mean, slope on accuracy (pp/pp, truths from DEV2
truths.json), region split (blind / 2~3 / 1~2 / tail, as T292) and per-model SE.

**Go / stop (DRI line):** per closure, **go** iff RMS_m(smooth) <= 0.70 x RMS_m(hard) AND |slope(smooth)| <= |slope(hard)|
+ 0.01. Otherwise **stop** for that closure. Overall go if either closure goes. I also report Fable's own secondary
line (mean offset halved; stop < 10% cut). Labels: the closure is a declared structural approximation; the Phi
smoothing and J are deductive (exact local map, model-free s); the oracle is diagnostic sampling.
Cap 1 h.
