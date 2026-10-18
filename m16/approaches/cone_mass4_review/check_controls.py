"""Independent analytic products and positive simplex controls of frozen author code."""
import hashlib,json,math,time,resource,warnings
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from . import author_snapshot as M
HERE=Path(__file__).parent
EXPECTED='6617ddd0a042d4af5fbc899e0542b9e31c1ee23f5ab8d47e5fb8ee2c005f05a6'
assert hashlib.sha256(Path(M.__file__).read_bytes()).hexdigest()==EXPECTED

def positive_mass(R,n=16):
    R=R/np.sum(R,axis=1)[:,None];assert np.min(R)>=0
    x,w=leggauss(n);x=(x+1)/2;w=w/2;u,v,z=np.meshgrid(x,x,x,indexing='ij');a,b,c=np.meshgrid(w,w,w,indexing='ij')
    y=np.stack([u,(1-u)*v,(1-u)*(1-v)*z,(1-u)*(1-v)*(1-z)],axis=-1).reshape(-1,4);weights=(a*b*c*(1-u)**2*(1-v)).ravel()
    points=y@R;det=abs(np.linalg.det(R));value=det*np.dot(weights,1/np.sum(points*points,axis=1)**2)/(2*math.pi**2)
    bound=det*(12*n+9)/(6*math.pi**2*3.**(n-1));return float(value),float(bound)

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,22));start=time.process_time();M._GL.clear();out=dict(source_sha256=EXPECTED,analytic=[],positive=[],invalid=[],author_costs_partial=True)
    def evaluate(kind,X,truth=None,bound=0):
        ops=M.Ops();before=set(M._GL);t=time.process_time();row=dict(interface=kind)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                value=(M.mass4 if kind=='normals' else M.mass4_rays)(X,ops)
            row.update(value=float(value),finite=bool(np.isfinite(value)),negative=bool(value<0))
            if truth is not None:row.update(truth=truth,reference_bound=bound,error=float(value-truth),passes=bool(np.isfinite(value) and abs(value-truth)<=1e-7+bound))
        except Exception as exc:row.update(failure_type=type(exc).__name__,failure=str(exc),passes=False)
        row.update(named_ops=ops.n,cpu=time.process_time()-t,new_gl_orders=sorted(set(M._GL)-before));return row
    for a in (-.9,0.,.8,1-1e-6):
        for b in (-.7,0.,.6):
            N=np.array([[1,0,0,0],[a,math.sqrt(1-a*a),0,0],[0,0,1,0],[0,0,b,math.sqrt(1-b*b)]],float);R=np.linalg.inv(N).T
            truth=(.25+math.asin(a)/(2*math.pi))*(.25+math.asin(b)/(2*math.pi))
            for kind,X in (('normals',N),('rays',R)):
                for transform in ('base','scale','permute'):
                    Y=X if transform=='base' else X*np.exp2([-20.,20.,-10.,10.])[:,None] if transform=='scale' else X[[3,1,0,2]][:,[2,0,3,1]]
                    row=evaluate(kind,Y,truth);row.update(rho1=a,rho2=b,transform=transform);out['analytic'].append(row)
    for eps in (.01,1e-6,1e-10):
        R=(np.eye(4)+.1*np.ones((4,4)));R[:,3]*=eps
        truth,bound=positive_mass(R);row=evaluate('rays',R,truth,bound);row.update(epsilon=eps,positive_reference_order=16);out['positive'].append(row)
    for kind in ('normals','rays'):
        row=evaluate(kind,np.zeros((4,4)));row['unsupported_singular_input']=True;out['invalid'].append(row)
    out.update(complete=True,all_valid_pass=all(r.get('passes',False) for r in out['analytic']+out['positive']),cpu=time.process_time()-start,gl_orders=sorted(M._GL),gl_construction_excluded_from_author_ops=True,maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('analytic','positive','invalid')},indent=2));print(json.dumps([r for r in out['analytic']+out['positive'] if not r.get('passes')],indent=2))
if __name__=='__main__':main()
