# A remaining fixed-rule obstruction after exact inner integration

This is a real-arithmetic statement about the frozen global row permutation
and the original wide product family, not a floating implementation result.

After placing the three orthant generators first, every outer interior node
has first orthant coordinate u>0. Thus q(u,v,s,w)>=u² for every inner w in[0,1],
independently of the angle between the final wedge rays. Its exact positive
inner integral is at most u^-5. A fixed finite set of strictly interior outer
nodes has a finite weighted Jacobian sum of these bounds, independent of the
wedge angle. Multiplying by |det R|, which tends to zero as the wedge angle
approaches pi, forces this fixed analytic-inner cubature estimate to zero.
The true product-cone probability tends to1/16.

Therefore analytic elimination of the inner coordinate alone cannot make this
particular fixed-interior outer rule uniformly accurate on the original wide
family. It reduces the tensor dimension but leaves an outer endpoint layer.
Exact splitting changes the parameterization and invalidates the premise of
the same collapsing determinant family. Adaptive outer rules, different charts,
further analytic elimination and rigorous local subdivision remain open.

Finite observed values are diagnostics of the frozen implementation; this
argument does not certify its floating arithmetic, any split-rule error, or
an impossibility for the whole analytic-inner method family.
