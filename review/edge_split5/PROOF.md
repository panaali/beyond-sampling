# Exact partition, independent of quadrature

Let independent rays r0,...,r4 form parent R, and certify the actual candidate v=a*r0+b*r1 with rational a,b>0. Its two children replace row0 or row1 by v. Unique nonnegative coordinates x=sum lambda_i*r_i show that child0 is b*lambda0<=a*lambda1, while child1 is b*lambda0>=a*lambda1. Their union is the parent, and their interior intersection is empty. Equality is a hyperplane and has zero Gaussian mass. Thus child probabilities add exactly, irrespective of the integrator.

Det(child0)=a*det(R), det(child1)=b*det(R). For the parent-interior dual functional w with w dot r_i=1, w dot v=a+b. The two section-volume ratios are a/(a+b), b/(a+b), adding to1 exactly. These geometric ratios are NOT Gaussian mass fractions.

Ordinary normalization may round the intended sum direction. We do not silently infer its exact span/positive coefficients: solve R^T*c=v in Fractions using the actual stored binary64 values, reject if any c2..4 is nonzero or c0/c1 is nonpositive. The product fixtures' coordinate plane makes this check feasible; general off-plane rounding can fail and remains unsupported. Exact additivity refers to the accepted actual child cones; reference angles refer to their actual stored rays.
