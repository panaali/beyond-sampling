"""Deterministic checks of ball normalization, surface Jacobian, and gate fiber."""
import json
from math import gamma,pi,sqrt
from pathlib import Path
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad


def main():
    start=time.process_time();balls=[];integrand_calls=0
    for r,k in [(1,2),(1,3),(2,3),(2,4),(3,4),(3,5),(3,6),(3,10)]:
        prefactor=gamma(k/2)/(gamma(r/2)*gamma((k-r)/2))
        vals=[]
        for power in (0,1,2):
            val,err,info=quad(lambda t:t**power,0,1,weight='alg',
                              wvar=(r/2-1,(k-r)/2-1),epsabs=1e-13,epsrel=1e-13,
                              full_output=True)
            integrand_calls+=info['neval'];vals.append(prefactor*val)
        exact=[1.,r/k,r*(r+2)/(k*(k+2))]
        balls.append({'rank':r,'k':k,'radial_moments':vals,'exact':exact,
                      'max_error':float(np.max(np.abs(np.array(vals)-exact)))})
    # Direct parameterized surface element, separate from the cofactor factor.
    theta=(np.arange(512)+.5)*2*pi/512
    sig2=np.array([.7,2.3]); v2=np.stack([np.cos(theta),np.sin(theta)],axis=1)
    derivative=np.stack([-sig2[0]*np.sin(theta),sig2[1]*np.cos(theta)],axis=1)
    density2=1/(2*pi*np.prod(sig2)*np.linalg.norm(v2/sig2,axis=1))
    ds=np.linalg.norm(derivative,axis=1)*2*pi/512
    surface2=[float(np.sum(density2*ds))]+[float(np.sum(density2*ds*(v2[:,i]*sig2[i])**2)) for i in range(2)]
    z,w=leggauss(32); phi=(np.arange(64)+.5)*2*pi/64
    zz,pp=np.meshgrid(z,phi,indexing='ij');rr=np.sqrt(1-zz**2)
    v3=np.stack([rr*np.cos(pp),rr*np.sin(pp),zz],axis=-1)
    dz=np.stack([-zz/rr*np.cos(pp),-zz/rr*np.sin(pp),np.ones_like(zz)],axis=-1)
    dp=np.stack([-rr*np.sin(pp),rr*np.cos(pp),np.zeros_like(zz)],axis=-1)
    sig3=np.array([.7,1.4,2.3]);jac=np.linalg.norm(np.cross(dz*sig3,dp*sig3),axis=-1)
    dens3=1/(4*pi*np.prod(sig3)*np.linalg.norm(v3/sig3,axis=-1))
    weights=w[:,None]*2*pi/64
    surface3=[float(np.sum(dens3*jac*weights))]+[float(np.sum(dens3*jac*weights*(v3[:,:,i]*sig3[i])**2)) for i in range(3)]
    fiber_mass=0.;fiber_first=0.
    for a,b,g in [(-pi/2,-pi/4,1.),(-pi/4,pi/4,.5),(pi/4,pi/2,0.)]:
        mass,_,info=quad(lambda th:g/pi,a,b,full_output=True);integrand_calls+=info['neval']
        first,_,info=quad(lambda th:g*np.sin(th)/pi,a,b,full_output=True);integrand_calls+=info['neval']
        fiber_mass+=mass;fiber_first+=first
    result={'ball_cases':balls,'max_ball_error':max(x['max_error'] for x in balls),
            'ellipse_surface_mass_second_moments':surface2,'ellipse_expected':[1.]+list(sig2**2/2),
            'ellipsoid_surface_mass_second_moments':surface3,'ellipsoid_expected':[1.]+list(sig3**2/3),
            'gate_fiber_mass':fiber_mass,'gate_fiber_first_submoment':fiber_first,
            'gate_fiber_expected_first':-1/(pi*sqrt(2)),
            'surface_parameter_nodes':512+32*64,'scalar_integrand_evaluations':integrand_calls,
            'model_forward_passes':0,'sampled_model_inputs':0,'cpu_seconds':time.process_time()-start}
    Path(__file__).with_name('projection_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='ball_cases'},indent=2))


if __name__=='__main__':
    main()
