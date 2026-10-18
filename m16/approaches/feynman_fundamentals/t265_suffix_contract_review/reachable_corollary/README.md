# T265 independent reachable same-summary proof review

## Question
Do two zero-start reachable prefixes of one fixed network have identical proposed summaries but unequal last-input success probabilities?

## Method
Written review of producer c4d81aaf under explicit assignment7bce65ef. Read the three frozen documents, check Git-byte and inherited hash bindings, and derive recurrence, winners, ranks and probability differences algebraically. No network, CDF, array, symbolic-control or candidate execution.

## Status
Written proof passes. Execution remains false. This is an independent review of a specific universal pointwise sufficiency counterexample, not a population or trained-model finding.

## Results
[inferred] For each p in {3/4,5/4}, the sequence (2,p−2,0) produces (p,2−p,0,0) after three steps: the negative second input removes2−p from the first coordinate and adds it to the second. Both coordinates are positive. Prefix order is strict with largest2 at index0 and second0 at index2. The readout is (p,2−p,3/2,−2), so its target2 margin is1/4 in both cases.

[inferred] Put s=p+y. When 0≤s≤2, the two positive coordinates sum to2, leaving class2 at3/2. It beats classes0 and1 precisely for1/2<s<3/2. Below that interval class1 wins, including its left tie; above it class0 wins, including its right tie. For s outside[0,2], the single positive coordinate beats its3/4 multiple and the nonpositive alternatives, so the same global classification holds. Class3 cannot win.

[inferred] Actual second-largest labels are2 for y≤0,3 for0<y≤2, and0 for y>2. At0 the prior index2 owns the second-place tie; at2 the prior index0 owns first place and the new index3 owns second. Because both chosen p lie strictly between1/2 and3/2, correctness is exactly (1/2−p,0]∪(2,∞). Therefore q(p)=1/2+Phi(p−1/2)−Phi(2), and the difference is Phi(3/4)−Phi(1/4)>0 by positivity of Gaussian density. Endpoint assignments are correct even though their continuous-law mass is zero.

[inferred] A single value assigned to the common summary cannot equal both probabilities; its worst absolute error over these two points is at least half their difference. This follows from the triangle inequality. Their prescribed prefixes have Gaussian probability zero; the argument does not refute almost-sure sufficiency or provide positive-measure error, bias or MSE bounds. It does not rely on choosing a conditional-probability version at null points: the q values here are the explicitly defined fresh-suffix kernel.

## Compute accounting
Only text reading, Git-byte comparisons and SHA256 hashing. No rational-control evaluator, forward pass, CDF approximation, RNG, target/model/checkpoint/array decode, fitting, geometry or mass call. The Gaussian difference remains symbolic.

## Claims
[inferred] The exact top-two/current-target-margin summary is not universally pointwise sufficient over reachable states of this synthetic network. The proof strengthens the earlier extension-only example without changing the completed symbolic suite or its scope.

## Alternatives not ruled out
Richer summaries (including a hidden coordinate), branch-specific injectivity, controlled approximations, signed integrated error and other weak-success representations remain open. No cost or compression benefit is measured.

## Failure modes
Dropping the permanent tie policy, interpreting two isolated states as a positive-probability population obstruction, or extending this example to trained weights would exceed the proof. Any future numerical or population phase needs its own scope and gates.
