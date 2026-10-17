"""Independent cosine-integral coefficients and positive endpoint-tail checks."""
import json
import time
from pathlib import Path
import mpmath as mp
import numpy as np
from chebyshev_positive import positive_tail_bound


def main():
    start=time.process_time();mp.mp.dps=70
    rows=[];calls=0;hyper_calls=0
    for lower in [.2,.95]:
        for nu in [1.,2.5,3.,5.]:
            a=mp.mpf(str(lower));b=mp.mpf(1);p=mp.mpf(str(nu))
            m=(a+b)/2;delta=(b-a)/2;s=mp.sqrt(a*b)
            r=(mp.sqrt(b)-mp.sqrt(a))/(mp.sqrt(b)+mp.sqrt(a));C=(mp.sqrt(a)+mp.sqrt(b))**2/4
            for N in [1,4,12]:
                def coefficient(k):
                    nonlocal hyper_calls
                    hyper_calls+=1
                    factor=1 if k==0 else 2
                    return factor*C**-p*r**k*mp.rf(p,k)/mp.factorial(k)*mp.hyp2f1(p+k,p,k+1,r*r)
                exact_coeff=coefficient(N)
                def integrand(th):
                    nonlocal calls
                    calls+=1
                    return (m+delta*mp.cos(th))**-p*mp.cos(N*th)
                cosine_coeff=2/mp.pi*mp.quad(integrand,[0,mp.pi/4,mp.pi/2,3*mp.pi/4,mp.pi])
                tail=a**-p-sum(coefficient(k) for k in range(N))
                bound=positive_tail_bound(lower,1.,nu,N-1)
                exact_special=None
                if nu==3:
                    alpha=m/s
                    exact_special=r**N/(s**3*(1-r))*(N*N+3*alpha*N+3*alpha**2-1
                        +(2*N+3*alpha)*r/(1-r)+r*(1+r)/(1-r)**2)
                rows.append({'lower':lower,'upper':1.,'nu':nu,'N':N,
                             'coefficient_relative_error':float(abs(abs(cosine_coeff)/exact_coeff-1)),
                             'positive_tail':str(tail),'bound':bound if np.isfinite(bound) else None,
                             'bound_over_tail':float(mp.mpf(bound)/tail) if np.isfinite(bound) else None,
                             'nu3_tail_relative_error':None if exact_special is None else float(abs(tail/exact_special-1))})
    ratios=[x['bound_over_tail'] for x in rows if x['bound_over_tail'] is not None]
    result={'cases':rows,'max_coefficient_relative_error':max(x['coefficient_relative_error'] for x in rows),
            'minimum_bound_over_tail':min(ratios),'maximum_bound_over_tail':max(ratios),
            'max_nu3_tail_relative_error':max(x['nu3_tail_relative_error'] or 0 for x in rows),
            'cosine_integrand_evaluations':calls,'high_precision_hypergeometric_evaluations':hyper_calls,
            'float_bound_evaluations':len(rows),'model_forward_passes':0,'sampled_model_inputs':0,
            'cpu_seconds':time.process_time()-start}
    Path(__file__).with_name('chebyshev_positive_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':
    main()
