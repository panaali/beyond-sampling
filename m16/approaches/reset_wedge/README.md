# T49 magnitude-dependent H2 reset wedge

## Question
Can a weight-derived reset region larger than two negative inputs reduce unresolved history for the trained H2T8 estimator? This lane is distinct from state-density transport, history moment closure and tail integration. Claim e30f4b9 was pushed before implementation. Claude remains DRI.

## Method and frozen algebra
Assume Wi=(w1,w2)>0 and Wh=[[-a,-b],[c,-d]] with a,b,c,d>=0. The initial hidden state ranges over the entire nonnegative quadrant, a stronger domain than the reachable set. Define Gamma=c*w1 when d=0; otherwise Gamma=max(0,c*w1-d*w2,c*max(w1-b*w2/d,0)). Proposed exact uniform-state reset set for a pair(x,y) is y<=-kappa*max(x,0), kappa=Gamma/w2.

Proof candidate: after the first input, maximize c*h1-d*h2 over old state(u,v)>=0. Increasing u cannot increase the objective, so u=0 attains the maximum. For x>0, remaining v dependence is piecewise linear with breakpoints w1*x/b and w2*x/d; its maximum occurs at v=0, v=w2*x/d, or the nonpositive/zero branch. For x<=0, h1=0 and the maximum is0. At the second input the second gate vanishes for every old state exactly when w2*y+Gamma*max(x,0)<=0. Since Gamma>=0 this also makes y<=0, forcing the first gate off. Equality resets exactly; there is no epsilon test.

For iid standard Gaussian inputs, q=atan(kappa)/(2*pi), reset-pair mass r=1/2-q. Let S_n be probability of no certified reset pair among n inputs (not absence of actual zero hidden states). Proposed exact recurrence S0=S1=1; S_n=.5*S_(n-1)+q*S_(n-2). The surviving positive final-input density is phi(x)*S_(n-1); negative x requires previous positive y>-x/kappa, whose surviving density is phi(y)*S_(n-2). For a last reset ending at j>=2, mass is r*U_(T-j), where U0=1,U1=.5 and the same recurrence holds. These masses plus S_T must sum1. kappa0 and kappa→infinity yield reset-on-any-negative and the original two-negative rule respectively.

## Frozen falsifiers and alternatives
1. The proposed wedge may only suffice, or may omit an optimizer/degenerate gate case. Independently maximize the piecewise-affine objective by enumerating all quadrant/activation-region vertices using rational arithmetic, compare synthetic positive/zero a,b,c,d cases, and construct violating states just outside the candidate boundary. No fitted coefficient.
2. Reset probabilities may be incorrectly multiplied despite overlapping pairs. Independently integrate the one-dimensional survival operator at small n using positive adaptive quadrature, plus exact limiting recurrences and total last-reset partition. No Monte Carlo estimator or trained accuracy sweep.
3. The enlarged wedge may add negligible probability at actual weights. Freeze H2T8 seed0 and the two existing relative-noise copies(.0025,.02; historical RNG1 protocol) before measuring coefficients/masses; preserve failure of sign assumptions. Verify checkpoint/model hashes and exact positive diagonal/input rescaling invariance.
4. Resetting hidden state does not erase input rank. The enlarged event couples the two discarded inputs; old independent-negative prefix rank laws cannot be reused. Keep all unresolved rank/history mass explicit. First phase delivers a reset theorem and mass inventory only, never a full accuracy claim.

## Compute accounting
Each cycle<=110CPU, one thread/2GB; primary compute lock for long runs, review lock only<=20CPU checks. Initial phase target<=20CPU. Rational operations/bit lengths, adaptive scalar callbacks, arithmetic/special-function proxies and measured CPU reported separately. No silent conversion of exact-rational operations into FLOPs. Source/review diagnostics are not production cost. No shared-source edits, no sampling advantage claim before independent review.

## Status and claims
Status: theorem/control phase complete; rank coupling remains open. T55 independently reviewed the algebra in f49d187, including the distinction between absent pair certificates and absent actual resets. DRI classification remains pending.

[inferred, independently reviewed] The wedge is the exact uniform-state two-step reset set under the stated signs. The overlap recurrence and last-certified-pair partition are exact in real arithmetic. [measured] All 972 rational synthetic cases passed, with 2,188 vertex evaluations and 10,208 direct reset checks; independent scalar quadrature at six slopes agrees through n3 within 2.23e-16, and all mass partitions sum to one. No random model inputs were used.

## Results
Run `flock /workspace/work/codex-review.lock env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python -m m16.approaches.reset_wedge.check` from the research worktree. Results and complete model/checkpoint hashes are in `results.json`. Total check CPU: 0.246920 seconds (imports excluded).

| Existing H2T8 copy | kappa | No certified pair, T8 |
| --- | ---: | ---: |
| seed0 | 0.08898349890 | 0.01100327750 |
| relative noise .0025 | 0.09046615772 | 0.01106174626 |
| relative noise .02 | 0.10085114517 | 0.01147610084 |
| two-negative certificate | infinity | 0.21484375000 |

[measured] This is a substantial reduction of the unresolved pair-certificate stratum, not yet an accuracy estimator. Exact coefficient setup reports a 30-rational-operation proxy and parameter/intermediate bit-size diagnostic (87/209/205 summed numerator+denominator bits); scalar mass assembly uses a declared 114-operation proxy and one atan. These are transparent conventions, not audited portable cost bounds. Floating probability evaluation is not roundoff-certified. Exact diagonal hidden rescaling by powers 2±40 and generic-vertex maximization passed on all three copies.

## Alternatives not ruled out and failure modes
Reachable-state restrictions could yield an even larger reset set. Unsupported weight signs need a different theorem. Small input weights can make kappa huge and gains negligible. Rank dependence, long surviving suffixes and conditional integration error may dominate even if resets improve. Exact algebra does not certify floating numerical evaluation; no method family is declared solved or dead here.

## Rank factorization checkpoint
Protocol `RANK_PROTOCOL.md`, implementation `rank.py`, independent controls `check_rank.py`. The last reset leaves the older iid prefix unrestricted; only a two-input wedge rank law remains. The three unnormalized pair-count masses are nonnegative and include the reset probability. On25 frozen slope/threshold cases, independent Y-first rectangle integration agrees within6.67e-16; explicit Bernoulli-bit enumeration agrees with binomial convolution, and1,024 deterministic event checks confirm the last-pair/future-positive identity. Total control CPU.010055 seconds,8,043 scalar callbacks, no random model inputs. See `rank_results.json`; adaptive error estimates and floating arithmetic remain uncertified. This is a rank primitive, not a full estimator.

## Two terminal strata
`terminal.py` computes last certified pair ending at T and T−1, preserving rank comparisons. `check_terminal.py` independently checks the first with exchangeability plus labeled-pair order statistics and the second by reversing integration order and using a beta tail. Exact-rational one-step readout dot products give label7 on all three H2T8 copies; zero hidden state gives label0 by first-index ties.

| copy | last pair T | last pair T−1 | partial accuracy | unresolved event mass |
| --- | ---: | ---: | ---: | ---: |
| base | .06912486824189505 | .06529653195477915 | .1344214001966742 | .271187441976275 |
| .0025 | .06909699972477873 | .06526891158197362 | .13436591130675235 | .27153857383164737 |
| .02 | .06890195958938533 | .06507568930312446 | .1339776488925098 | .273995305173665 |

Run the same locked/thread-limited Python command as above with module `m16.approaches.reset_wedge.check_terminal`. Total CPU .053997 seconds excluding imports; each copy uses777 outer scalar callbacks and38,241–38,262 rank-kernel callbacks, plus rational readout setup and beta/CDF operations. Both production and independent-reference work are included in those diagnostic totals. No portable cost or network forward-equivalent claim. Independent formulas differ at most3e-17. All omitted strata remain explicit; the real-arithmetic partial sum is a lower contribution, but ordinary adaptive error estimates/roundoff do not make the saved floating result a formal enclosure. This is not a full network accuracy estimate.

Wide-slope hardening found and preserved a numerical failure: kappa1e8 caused the original adaptive integral to miss a narrow boundary layer, losing1.59e-9 with reported error0 (`wide_scale_before.json`). Explicit breaks at1/kappa,3/kappa,8/kappa fix the four frozen scale controls to3.34e-16 partition error (`wide_scale_after.json`). Existing25 rectangle checks and all three terminal contributions pass unchanged after the fix; repeat CPU.009310 and.058636 seconds respectively. This is a quadrature setup failure and bounded rescue, not a failure of the algebra or a general numerical certification.

## Generic conditional ranks and two-input suffix
Scope extension86a43c1 and `SUFFIX_PROTOCOL.md` froze this phase. `conditional.py` handles every old, reset-pair and suffix target label;24 small count/threshold configurations sum to the reset mass within9.99e-16 and preserve exchangeability. CPU.094497 seconds. Fixed suffix ties are rejected; this does not alter a continuous outer integral.

`two_suffix.py` uses exact Fraction gate/readout roots for h=v*ReLU(Wh*Wi+Wi*z), v>0,z>−kappa. On the base model,112 pair-root checks produce21 regions, all with suffix predictions6/7. Positive-threshold Owen-T identities match independent positive rank integrals within1.39e-16. The initial Laguerre8/16/32 radial rule is unconverged (.03845996/.03825614/.03819176), preserved in `two_suffix_results.json`. Direct-radius positive Legendre8/16/32/64 gives .03782518859/.03815795516/.03815806053341955/.03815806053341946. Explicit radial tail allowance1.63e-15 covers truncation only; refinement agreement is not a quadrature certificate. At order64 the full integral/setup takes .010156 seconds,86,016 Owen-T calls,172,032 normal-CDF calls and the listed rational setup, with zero negative clamps. This is a callback ledger, not full portable operations. All four direct-radius runs total.021636CPU seconds excluding imports.

The base model's three resolved strata contribute .17257946073009367 and leave .14285571617911716 event mass unresolved. No total accuracy, MSE advantage or solved-rung claim. T55 independently reviewed rank and first two terminal formulas in23deb0d; final two-suffix source review is pending. Commands: locked/thread-limited Python modules `check_conditional`, `check_two_suffix`, `check_radius` under this package. Original failures remain visible; no mass renormalization or fitted correction is used.
