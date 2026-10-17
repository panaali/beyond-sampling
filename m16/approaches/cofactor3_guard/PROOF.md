# Frozen error-enclosure theorem

Write the stored3x3 matrix as rows(a,b,c),(d,e,f),(g,h,i). Every operation below is one separately rounded binary64 addition, subtraction or multiplication, with round-to-nearest and no fusion/reassociation.

```
p0=e*i; p1=f*h; m0=p0-p1; q0=a*m0
p2=d*i; p3=f*g; m1=p2-p3; q1=b*m1
p4=d*h; p5=e*g; m2=p4-p5; q2=c*m2
r=q0-q1; Dhat=r+q2
```

The permanent approximation Phat uses exactly this tree with absolute inputs, replacing all five subtractions by additions (three minor subtractions plus the one outer subtraction; the final sum is already an addition, so four replacements). It has9 multiplications and5 additions, while the determinant has9 multiplications,4 subtractions and1 addition. Each exact triple term has at most five rounding factors along its path to the result.

Every nonzero input is in[2^-100,2^100] in absolute value. Stored inputs lie on a lattice no finer than2^-152. Products, minor differences, triple products and final sums stay finite and normal or zero: the three-input product lattice is no finer than2^-456, far above the smallest normal exponent, and absolute intermediate magnitudes are bounded by6*2^300. Rounding to a binary64 result cannot introduce a finer lattice than the inputs to that stage. Thus the usual relative-error model applies to every nonzero intermediate; exact zero results introduce no error.

For the exact positive permanent P, the standard fixed-expression expansion gives

```
|Dhat-D| <= gamma5 P,       gamma5 = 5u/(1-5u)
Phat >= (1-u)^5 P,         u = 2^-53.
```

The exact rational inequality8(1-u)^5(1-5u)>5 implies gamma5 P <=8u Phat. Multiplication by8u=2^-50 is exact here: if P is nonzero its positive triple terms are at least2^-300, so the scaled result remains normal. If Phat=0, all exact positive triple terms vanish and D=0.

Set E=2^-50 Phat. The real endpoints Dhat-E and Dhat+E are enclosed by nextafter(fl(Dhat-E),-infinity) and nextafter(fl(Dhat+E),+infinity). Endpoint rounding is therefore included explicitly. Preserve signed-zero input cases. When E=0, return the exact zero interval after confirming Dhat==0.

Finite inputs outside the guard are not covered by this lemma. The fallback computes D exactly using rational representations of the stored floats and encloses the correctly converted result with directed neighboring floats, retaining overflow/underflow endpoints. Nonfinite inputs have no real determinant certificate and return unknown. The fallback's exact computation and large integer bit lengths are separate cost categories.

This theorem certifies one stored3x3 determinant, not real unrounded upstream coefficients, signs when zero lies in the enclosure, relative accuracy, final mass error, native implementation cost or the ARC challenge.
