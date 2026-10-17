# Fractional-power Chebyshev coefficient majorant

For 0<a<b, set K=((sqrt(a)+sqrt(b))/2)^2 and rho=(sqrt(b)-sqrt(a))/(sqrt(b)+sqrt(a)). On the unit circle z=exp(i theta), q=(a+b)/2+(b-a)cos(theta)/2=K(1+rho*z)(1+rho/z).

Write c(nu,j)=(nu)_j/j!. The binomial product gives the coefficient at Fourier index k>=0 as (-1)^k K^-nu rho^k sum_{j>=0} c(nu,j+k)c(nu,j)rho^(2j). These series converge absolutely because rho<1. Each summand has the same sign. The recurrence c(nu,j+1)=c(nu,j)(nu+j)/(j+1) proves c(5/2,j)<=c(3,j) for every j. Consequently the magnitude of every Fourier coefficient at5/2 is at most sqrt(K) times the corresponding magnitude at3. Pairing positive/negative indices gives the same inequality for Chebyshev coefficients, including its factor2.

For the Chebyshev polynomial truncated at degree N, its uniform remainder is bounded by the sum of coefficient magnitudes past N. Thus E_(5/2),N <= sqrt(K)*E_3,N, where E_3,N is the reviewed integer-power tail implemented by Directed.tail(a,b,3,N). For a=b both tails are zero. Expanded outward endpoints are used consistently for BOTH factors and the interval to which the polynomial applies.

The five-dimensional Duffy rule has positive weights, simplex volume1/24, and exactly integrates a polynomial in q of degree N=n-2: q is at most quadratic in each Duffy variable, while its Jacobian has coordinate degrees3,2,1,0, so 2N+3<=2n-1. The probability prefactor is3|det R|/(8pi^2). Integral and quadrature errors relative to the same polynomial each contribute at most simplex-volume times E. Hence the ideal-rule probability error is at most |det R|*sqrt(K)*E_3,N/(32pi^2). There is no extra hidden factor2; it is already in this prefactor and in the definition of the Chebyshev coefficient tail.

For independent rational controls with rational sqrt(a),sqrt(b), let s=sqrt(ab), B=(a+b)/(2s), r=rho and m=N+1. The integer3 coefficient magnitude at k>=1 is r^k/s^3 * (k^2+3Bk+3B^2-1). Summing the geometric moments yields E3=r^m/s^3 * [(m^2+3Bm+3B^2-1)/(1-r)+(2m+3B)r/(1-r)^2+r(1+r)/(1-r)^3]. This exact rational formula tests the nu3 specialization independently of the implementation's polynomial/Eulerian recurrence.

This is an ideal quadrature truncation bound conditional on valid radial range and determinant inputs. Stored coefficients, numerical geometry, floating quadrature nodes/weights, kernel evaluation and final addition need separate analysis. A majorant may be loose and may even be worse than the old centered-binomial bound on nearly singular original cones. No monotonic improvement or order tuning is assumed.
