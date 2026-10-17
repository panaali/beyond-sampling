# T67 direct target-first transition review

T55 read-only review of `direct_first_cycle/core.py` and `driver.py`, 2026-10-17. This is a fixed finite-grid discriminator, not an exact network integration.

Conditional on target J=0 and value u being second-largest, exactly one of the remaining T-1 inputs is above u. The next input is above with probability 1/(T-1), after which zero exceedances remain, or below with probability (T-2)/(T-1), after which one remains. For T4 the implemented 1/3 high/need0 plus 2/3 low/need1 is correct. The preceding backward step at stage2 has two inputs remaining, so its 1/2 mixture is also correct. Terminal values are evaluated at stage3. The moving state h1=ReLU(Wi*u) is exact for this target-first path.

On each sign half of u, h1 is linear, and second-step gate/grid crossings have the affine form x=(grid-b*u)/Wi. The code includes zero gates through the zero grid node, all stage2 grid coordinates, both input support endpoints, and the rank line x=u. All pairwise crossings, u=0 and target endpoints therefore suffice for this direct transition's finite-grid topology in exact arithmetic. Existing later-stage knots cover the continuation coefficients. Rounded intersections/branch topology and inverse CDF remain numerical qualifications.

The stage2 cached matrices depend on stage and q, not the target label, so sharing their cache is valid. Values are combined in the correct remaining-exceedance columns. Conditional kernels and interpolation are positive contractions in exact arithmetic; the stated accumulated quadrature-only allowance can conservatively carry prior operator error. It does not bound spatial interpolation or floating arithmetic. Direct and baseline variants use the same union of panels, so the comparison is not a hidden panel-refinement experiment.

Cost counters remain partial: direct h1 multiplications/maxima and repeated normal-CDF setup are not separately charged in the displayed event categories. The driver already declares portable_flops_certified false. Preserve that status rather than converting event counts into a complete cost claim.

Removing one interpolation step need not monotonically reduce the final accuracy error: prior interpolation errors can cancel within a label or between label contributions. The frozen per-label geometric diagnostic is a suitable distinct test after the observed aggregate deterioration. It should be reported with its own numerical reference error and shared-geometry assumptions, without selecting a correction from truth.
