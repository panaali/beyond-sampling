# Seven analytic controls, not a trained estimator

The retained top two inputs occupy positions zero and one. The last input X is
independent standard Gaussian conditional on X<=b, where b is their minimum.
All seven models have Wi=(1,-1,0,0), Wh=cI, positive first retained input and
nonnegative second retained input. Immediately before the last input the state
is (c*x0+x1,0,0,0). The final two nonzero states are (S+X)+ and (-X)+, where
S=c*c*x0+c*x1>0. With scores(hminus,hplus,0), target1 wins precisely X>-S/2.
At equality class0 wins. Swapping scores reverses the open-event correctness;
identical rows always predict class0. These separately derived statements
validate the generic final-gate and readout-root partition.

For every competitor, the candidate obtains the exact anchor0 margin and last
input global Lipschitz allowance L=abs(Wo_target-Wo_rival) dot abs(Wi). It uses
the reviewed scalar mean precheck and a preregistered lambda=abs(margin)/L².
Positive margins give a crossing/failure bound; negative margins give an upper
bound on the necessary crossing. A union bound provides the correctness lower
bound, while the minimum necessary-event bound provides its upper bound.
L=0 handles first-index constant ties exactly. Nonzero-L zero margins return
unknown. A vacuous mean precheck means only this anchor/global-envelope family
cannot improve on one; no claim about the actual error or other bounds follows.

The conditional CDF probabilities and candidate bounds are finite MP diagnostics,
not directed enclosures. Separately, for retained(4,3), c1, the earlier rival0
dominates rival2 everywhere (hminus>=0). Therefore its sole strict margin suffices.
Its anchor margin is7 and L2. Phi(3)>=1/2 and the ordinary absolute-normal MGF
give E[exp(2lambda|X|)|X<=3]<=4exp(2lambda²). At lambda7/4 this proves error
<=4exp(-49/8). Since exp(z)>=sum_{j=0}^{10}z^j/j! for z>0, the exact rational
upper4/partial is below1/100. This is a real-theory conditional certificate for
one synthetic model. It does not require trusting the finite CDF values.

Each partition cell stores exact final preactivation and score coefficients,
rational endpoints/probe, gate mask and first-index winner. Endpoint plus
recession-sign checks prove affine sign ownership over the entire open cell;
all gate/readout boundaries and the truncation endpoint get exact direct checks.
Boundary values have zero conditional Gaussian mass; permanent ties do not.

Accounting separates seven candidate anchor forwards from validation prefix,
cell and boundary forwards. Exact arithmetic, MP/CDF and partition preparation
are not assigned a hardware-equivalent price. No random inputs, trained models,
reference quadrature, common harness run, population mass or target scaling claim.
Alternative explanations: an apparently good result may exploit this toy's
dominated rival (generic all-rival bound is reported separately); a bound may
remain vacuous despite moderate true accuracy (the frozen vacuous case tests it).
No lambda tuning or performance-dependent case selection is permitted.
