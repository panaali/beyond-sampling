# Frozen source contract

All statements refer to60967f893fb649ca9933bd026ca76c3c19402e15, not a new run or altered shared estimator.

- core.py::_face_tables uses160001 grid points over[-8,8]. With e=exp(-u²)/sqrt(pi), it forms d12=e*Phi(u)^(T-2), d23=e*(1-Phi(u))*Phi(u)^(T-3). The cumulative sum is multiplied by grid spacing. The returned inverse-CDF abscissae are(c-c[0])/(c[-1]-c[0]), while returned normalization is c[-1]. These are distinct normalizers.
- Position counts are n12=T(T-1)/2 and n23=n12(T-2); sourceZ is n_kind*c[-1]. Z is a co-area weight, not the probability of choosing kind in a uniform two-kind mixture.
- face_points uses U[:,0] for a uniform configuration index and U[:,1] for np.interp through the stored inverse CDF. Remaining independent uniform coordinates select a conditional maximum for kind23 and lower coordinates for both kinds. ndtri arguments are clipped to[1e-300,1-1e-16]; tau is capped at8.5. These modifications are model-independent but remain differences from the ideal law/integrand.
- tail.py::label_point calls walk_one and derives correctness components before it sets ub. If there are no components, it returns None. On nonempty components, ub=0 foru<=0,1 for0<u<=.25,2 for.25<u<=.5,3 foru>.5, with u=y[a]. The same bin can be computed on every generated face before walk_one. This is a proposed extension of the available input label; it does not retroactively fill omitted saved labels.
- analyse.py defines coarse=(kind,ub,wt,min(nflips,3)); wt and nflips depend on the largest wrong component and its gate transitions. Dropping those fields defines a new coarser partition, not an equivalent label or proven variance-preserving replacement.
- The original tail calculation uses separate finite Sobol rules per kind, then selects a combined global tail. The T230 analytic mixture chooses kind uniformly and weights X=Z_kind F. Neither construction makes Z a kind-selection probability, and neither turns selected-tail label frequency into full-face bin probability.

Only text was read/copied. There is no saved grid, generated face or mass output in this task. In particular, this phase does not assert that a specific runtime's floating CDF array satisfies the formal grid assumptions below; that requires a separately frozen implementation/control check.
