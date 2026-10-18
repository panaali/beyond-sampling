# T247 sole analytic-grid control result

Raw author commit `6d2db5f4` is published. The immutable gate `89c208e1` was
published through `9bff1348` before execution, following h4 review `006ca7d5`
(shared `2b83e21d`) and board execution scope `e136bc00`. Independent saved
review is pending; these are author observations until that review completes.

All four frozen cases passed without changing thresholds or rerunning. Three
2-interval synthetic controls retain the original and alternative central masses:
constant 1/2 versus1/2, symmetric peak about2/3 versus1/2, asymmetric about1/3
versus1/3. The asymmetric example remains asymmetric after the correction.

For the unchanged source's analytic T2 grid, the exact-real inverse-linear
pushforward of the saved binary64 CDF has:

| Stored-array diagnostic | Original | Local trapezoidal |
|---|---:|---:|
| P(Y<=0) minus1/2 | 2.8209479203455956e-5 | 3.319566843629218e-14 |
| Mean | -5.0000000286357386e-5 | -3.3825081341597526e-13 |
| Maximum CDF reflection defect | 5.641895840691191e-5 | 8.57467769972315e-14 |
| Zero-increment CDF intervals | 27,891 | 27,891 |
| Positive-increment intervals | 132,109 | 132,109 |
| Exact total interval mass | 1 | 1 |

The underlying stored grid is not exactly reflection-symmetric: maximum
`|g_j+g_(N-j)|=1.7763568394002505e-15`; maximum paired density difference is
`8.881784197001252e-16`. Thus these observations do not simply assume the
ideal symmetric premises of T238. The reported law contractions use exact
integers/Fractions; the table displays rounded decimal diagnostics.

The source co-area Z12 is `0.9999999999976183`; the separately computed local
trapezoidal normalizer is `0.9999999999976042`. Rounded library `erf(8)` returns
1. Their differences are about-2.38165e-12 and-2.39575e-12 respectively. This
rounded reference is not an interval certificate, and centering has not repaired
normalization. Source first spacing is retained as `0x1.a36e2eb1c0000p-14`.
The two normalizers are not silently substituted for one another.

Eighteen fixed `np.interp` outputs are archived separately. Flat CDF intervals
have zero continuous latent mass, while finite endpoint handling can still affect
individual program inputs. Neither these probes nor the continuous-law
contractions certify a finite Sobol/RNG law, ideal Gaussian integration, a
trained face integrand, target accuracy or an estimator improvement.

## Accounting and preservation

Main-scope author CPU1.075249376s. Actual recorded soft/hard CPU limits20/20s;
soft/hard address-space limits2,147,483,648/2,147,483,648bytes. One thread and
review lock, no retries. All15 durable events and raw stage/final archives remain.
Original source grid/function once; three synthetic original cumsums; four
local alternatives;18 finite interpolation probes; eight exact stored-law
contractions. Source ndtr/exp are each one vector call. Copies, setup, exact
integer arithmetic and I/O are research overhead; no full native operation
price, source speed comparison, model/face_points/RNG call or cost win.

## Scope and remaining alternatives

This verifies a narrow stored-law centering problem and tests one local rescue.
It does not exhaust the tie-slicing method. Better accumulation, interval-width
normalization, tail representations, asymmetric higher-T laws, endpoint rules
and actual integrand error remain separate questions. No shared source is
modified or adopted. Further numerical work needs a new frozen scope and gate.
