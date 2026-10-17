"""High-precision quadrature-bound check and separate float-roundoff report."""
import json
from pathlib import Path
import time
import mpmath as mp
import numpy as np
from numpy.polynomial.legendre import leggauss


def main():
    started=time.process_time();mp.mp.dps=100
    n=4;nodes,weights=mp.gauss_quadrature(n,'legendre')
    xn,wn=leggauss(n)
    polynomials=[('linear',lambda t:mp.mpf('.5')+mp.mpf('.4')*t,mp.mpf('.9')),
                 ('endpoint_square',lambda t:(1+t)**2/4,mp.mpf(1)),
                 ('center_peak',lambda t:1-t*t,mp.mpf(1))]
    rows=[];integrand_calls=0
    for center in ['0','6','-6']:
        c=mp.mpf(center)
        for halfwidth in ['.01','.0001','.00000001']:
            h=mp.mpf(halfwidth);M=abs(c)*h+h*h/2;phic=mp.exp(-c*c/2)/mp.sqrt(2*mp.pi)
            for name,P,pmax in polynomials:
                def integrand(t):
                    nonlocal integrand_calls
                    integrand_calls+=1
                    return h*P(t)*mp.exp(-(c+h*t)**2/2)/mp.sqrt(2*mp.pi)
                reference=mp.quad(integrand,[-1,0,1])
                high=sum(w*integrand(x) for x,w in zip(nodes,weights))
                bound=4*h*phic*pmax*mp.exp(M)*M**(n-1)/mp.factorial(n-1)
                cc,hh=float(c),float(h)
                if name=='linear':
                    pp=.5+.4*xn
                elif name=='endpoint_square':
                    pp=(1+xn)**2/4
                else:
                    pp=1-xn*xn
                ordinary=hh*float(wn@(pp*np.exp(-(cc+hh*xn)**2/2)/np.sqrt(2*np.pi)))
                higherr=abs(high-reference);floaterr=abs(mp.mpf(ordinary)-reference)
                rows.append({'center':center,'halfwidth':halfwidth,'polynomial':name,
                             'reference':str(reference),'quadrature_bound':str(bound),
                             'high_precision_error':str(higherr),'high_error_over_bound':float(higherr/bound),
                             'float_absolute_error':float(floaterr),
                             'float_relative_error':float(floaterr/reference),
                             'float_exceeds_quad_only_bound':bool(floaterr>bound)})
    result={'cases':rows,'max_high_error_over_bound':max(x['high_error_over_bound'] for x in rows),
            'max_float_relative_error':max(x['float_relative_error'] for x in rows),
            'float_errors_exceeding_quad_only_bound':sum(x['float_exceeds_quad_only_bound'] for x in rows),
            'high_precision_integrand_evaluations':integrand_calls,
            'float_quadrature_nodes':len(rows)*n,'model_forward_passes':0,'sampled_model_inputs':0,
            'cpu_seconds':time.process_time()-started}
    Path(__file__).with_name('quadratic_gaussian_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':
    main()
