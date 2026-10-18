# Frozen analytic controls, before target enumeration

Claim abeaa86d authorizes source preparation; this phase executes only after the published independent 910 source/resource gate is pinned in control_gate.json. No target weights are loaded. core.py, controls.py and this document are hash-pinned in control_protocol.json. Exactly 18 controls execute once: seven geometric, eight small networks, three API rejections. Existing output/progress/temporary files prevent rerun; every control attempt is durable before its calls, every result before continuing. Exceptions retain the partial row, exact cones, ledger and traceback. External termination may leave a running row, never an inferred pass.

## Geometry and expected answers

The positive 3D orthant is clipped by (1,1,-1): the ON cone has rays e1,e2,e1+e3,e2+e3; OFF has e3,e1+e3,e2+e3. Additional forms (1,0,0),(-1,0,0),(0,0,0),(1,1,1) exercise one-sided, boundary-only and exact-zero ownership. A positive halfline extruded with forms (2,3) and (2,-3) gives respective boundary rays (3,-2),(3,2), and the signed vertical rays. Every returned cone is certified; an absent boundary-only branch is not a region.

## Network controls

Five networks through depth 3 have exactly 2,4,8 histories: (1) one neuron Wi=1,Wh=0,D=1; (2) one neuron Wi=-2,Wh=3,D=2; (3) Wi=(1,-1),Wh=0; (4) Wi=(1,1), both Wh rows (1,-1), giving cancellation; (5) Wi=(0,1),Wh=((0,1),(0,0)), testing a later pivot and zero forms. For one neuron every prior open history has both next signs because the fresh coefficient is nonzero. Networks 3/4 reduce to the sign of the fresh coordinate; network 5 additionally copies the preceding positive state and adds no fresh split beyond its second neuron.

Network 6 is Wi=(1,-1),Wh=((1,2),(3,5)),D=1 through depth 2: there are two halfplanes at depth 1 and three sectors each at depth 2. In coordinates x0=sign*u with u>0, exact fresh-coordinate slopes are (-1,3) and (-2,5), so counts are 2,6. Networks 7/8 apply Wi scaling by +2 or Wi sign reversal. All history/ray sets and hidden coefficients must obey the corresponding exact scaling/reflection invariance. No sampled directions replace full cones. Three invalid inputs (all-zero Wi, nonpositive denominator, malformed Wh) must reject, not report an empty enumeration.

## Resources, accounting, interpretation

One review-lock invocation; OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1; internal16 CPU seconds, external20 CPU seconds; ulimit -v2097152 KiB. No NumPy, estimator, model, reference, quadrature, Monte Carlo or RNG calls. Exact cone construction, certificate work and serialization time are charged; partial named integer counts are not full native cost and are not float FLOPs. Python gcd/division/powers/comparisons/hash/control and bit complexity are not converted into an instruction count. Peak retained integer size and process CPU/RSS are recorded.

Two alternatives are tested separately: count agreement could conceal a ray/constraint defect (exact analytic ray sets plus full cone certificates), or valid witnesses could conceal missing histories (analytic counts plus the complete-ray induction and different-code saved review). These finite controls do not alone prove all inputs; target-byte provenance, planar independent slope check and separately scoped deeper enumeration remain required. Neither a resource cap nor count discrepancy is a broad impossibility result.

Command after gate publication, from repository root:

    flock /workspace/work/codex-review.lock bash -c 'ulimit -t 20; ulimit -v 2097152; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; /workspace/clean-neurons/.venv/bin/python review/target_prefix_recount/controls.py'
