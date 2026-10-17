"""Opposite-order direct integration checks the supported cut matrix."""
import json
from pathlib import Path
import time
import numpy as np
from numpy.polynomial.legendre import legmul,legint,legval
from scipy.integrate import quad
from space import Space,PHI0,WIDTH,unit

HERE=Path(__file__).resolve().parent


def direct_entry(normal,ni,mi,nj,mj):
    v=unit(normal)
    if v[2]<=0:raise ValueError('independent primitive assumes positive axial component')
    a=np.zeros(ni+1);a[-1]=np.sqrt(2*ni+1)
    b=np.zeros(nj+1);b[-1]=np.sqrt(2*nj+1)
    antiderivative=legint(legmul(a,b))/2
    def value(phi):
        h=v[0]*np.cos(phi)+v[1]*np.sin(phi)
        lower=-h/np.hypot(h,v[2])
        radial=legval(1.,antiderivative)-legval(lower,antiderivative)
        return radial*np.exp(1j*2*np.pi*(mj-mi)*(phi-PHI0)/WIDTH)/WIDTH
    re,er,info=quad(lambda p:value(p).real,PHI0,PHI0+WIDTH,epsabs=1e-12,epsrel=1e-12,full_output=1)
    im,ei,infoi=quad(lambda p:value(p).imag,PHI0,PHI0+WIDTH,epsabs=1e-12,epsrel=1e-12,full_output=1)
    return re+1j*im,er+ei,info['neval']+infoi['neval']


def main():
    start=time.process_time();out=dict(roots=[],cuts=[],entries=[])
    s=Space(8,4,64);fine=Space(8,4,128)
    root=s.one();out['root_mass_total']=12*s.mass(root)
    assert out['root_mass_total']==1.
    for name,v in [('near_wall',[1.,0.,0.]),('bisector',[-np.sqrt(3),1.,0.])]:
        k=s.matrix(v);error=float(np.max(abs(k-np.eye(s.size))))
        out['roots'].append(dict(name=name,identity_error=error));assert error<1e-12
    normals=[[0.,0.,1.],[.6,-.7,.2],[1.,.1,.7],[-.1,.2,1e-6]]
    for v in normals:
        k=s.matrix(v);negative=s.matrix(-np.array(v));refined=fine.matrix(v)
        eig=np.linalg.eigvalsh(k)
        row=dict(normal=v,complement_error=float(np.max(abs(k+negative-np.eye(s.size)))),
                 hermitian_error=float(np.max(abs(k-k.conj().T))),
                 minimum_eigenvalue=float(eig.min()),maximum_eigenvalue=float(eig.max()),
                 refinement_error=float(np.max(abs(k-refined))))
        out['cuts'].append(row)
        assert row['complement_error']<1e-12 and row['hermitian_error']<1e-12
        assert eig.min()>-1e-9 and eig.max()<1+1e-9
        assert row['refinement_error']<1e-8
    for v,ni,mi,nj,mj in [([.6,-.7,.2],2,-2,3,1),([1.,.1,.7],4,2,1,-3),([.2,-.1,.8],0,0,0,0)]:
        numerical=s.matrix(v)[ni*s.nang+mi+s.M,nj*s.nang+mj+s.M]
        direct,error,calls=direct_entry(v,ni,mi,nj,mj)
        row=dict(normal=v,indices=[ni,mi,nj,mj],matrix=[numerical.real,numerical.imag],
                 independent=[direct.real,direct.imag],error=float(abs(numerical-direct)),
                 independent_quadrature_error=error,independent_integrand_calls=calls)
        out['entries'].append(row);assert row['error']<1e-9
    out.update(cpu_seconds=time.process_time()-start,work=dict(s.work),fine_work=dict(fine.work))
    (HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(cpu_seconds=out['cpu_seconds'],max_entry_error=max(r['error'] for r in out['entries']))))

if __name__=='__main__':main()
