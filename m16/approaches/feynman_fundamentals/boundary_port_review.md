# T20 completed stochastic boundary-line port review

T55 read-only source review of `boundary_estimator.py` and pinned zone/allocator sources, 2026-10-17. No new sampling or accuracy run.

The A/B geometry already uses the unit direction with entries +1/sqrt(2), -1/sqrt(2). Therefore, unlike the radial tube adapter, no additional sqrt(2) coordinate conversion is needed. Intersecting the original |t| band with the mask gives upper max(min(tau,mhi),0) and lower mlo. The completed walker selects only upper>lower. A has positive-side target a and B positive-side target b, so `top=(tp==b)` selects the corresponding target convention correctly. Multiplication by the original positive geometry weight occurs once.

For constant first-index output zero, the exact fixed-line value is weight times `(1[tn!=0]+1[tp!=0])*(Phi(upper)-Phi(lower))`, with zero on an empty band. This is the correct independent control for completed lines; it is not the whole stochastic estimator's true accuracy. Replacing the old partial line plus random remainder changes the RNG stream and conditional variance. The wrapper correctly declares this as a completed-line stochastic variant, not bitwise equivalence to the legacy estimator.

R-zone rejection charges every generated candidate and its classification, including rejected and overshoot rows, rather than discounting generation by n/got. Only selected n rows receive full forward/rank charges. The explicit candidate cap raises instead of returning an incomplete estimate. Setup, geometry and opaque random/special-function operations remain numerical proxies, so a complete list of cost categories is not a portable bound.

The inherited two-stage allocator reuses pilot observations in means whose sample counts depend on those observations. No generic unbiasedness claim follows from conditional line integration alone. The source explicitly retains that qualification. Completing lines and improving bookkeeping does not independently validate the outer zone normalizers, finite inverse tables, adaptive sampling inference or a network-accuracy score.
