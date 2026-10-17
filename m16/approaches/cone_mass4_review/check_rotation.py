"""Finite geometry-only rotations for unresolved original author failures."""
from fractions import Fraction
from itertools import combinations
import hashlib,json,time,resource
import numpy as np
from .check_controls import positive_mass,HERE
from .replay import CAPTURES

def rotate_to_orthant(R):
    attempts=0
    for i,j in combinations(range(4),2):
        for denominator in (16,8,4,2):
            t=Fraction(1,denominator);c=(1-t*t)/(1+t*t)
            for sign in (1,-1):
                s=sign*2*t/(1+t*t);assert c*c+s*s==1
                attempts+=1;Y=R.copy();Y[:,i]=float(c)*R[:,i]-float(s)*R[:,j];Y[:,j]=float(s)*R[:,i]+float(c)*R[:,j]
                if np.all((Y.min(0)>=0)|(Y.max(0)<=0)):
                    Q=np.eye(4);Q[i,i]=Q[j,j]=float(c);Q[i,j]=float(s);Q[j,i]=-float(s)
                    return Y,dict(pair=[i,j],t=str(t),c=str(c),s=str(s),attempts=attempts,rotation_arithmetic=24*attempts,orthogonality_residual=float(np.max(abs(Q.T@Q-np.eye(4)))))
    return None,dict(attempts=attempts,rotation_arithmetic=24*attempts)

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,22));start=time.process_time()
    guard=json.loads((HERE/'guard_results.json').read_text());refs=json.loads((HERE/'independent_results.json').read_text())['rows'];out=dict(rows=[],source_modified=False,full_accuracy_certified=False,portable_cost_certified=False)
    for corpus in guard['corpora']:
        kind=corpus['kind'];path=HERE.parent/'class_region_t4'/(kind+'_exact_mass_calls.npz');assert hashlib.sha256(path.read_bytes()).hexdigest()==CAPTURES[kind]
        rays=np.load(path)['rays'];values=np.load(HERE/(kind+'_guard_values.npz'))['values'].copy()
        for failure in corpus['unresolved']:
            i=failure['index'];before=time.process_time();Y,meta=rotate_to_orthant(rays[i]);row=dict(kind=kind,index=i,original=values[i],**meta)
            if Y is None:row['unresolved']=True
            else:
                val,bound=positive_mass(abs(Y),20);ref=next(r for r in refs if r['kind']==kind and r['index']==i)
                values[i]=val;row.update(value=val,integration_only_bound=bound,mp_reference=ref['reference'],difference=val-ref['reference'],fallback_integrand_arithmetic=312000)
            row['cpu']=time.process_time()-before;out['rows'].append(row)
        out[kind]=dict(estimate=1-float(values.sum()),signed_reference_error=1-float(values.sum())-.477013125994119,negative_count=int(sum(values<0)))
        np.savez_compressed(HERE/(kind+'_rotation_values.npz'),values=values)
    out['cpu']=time.process_time()-start
    (HERE/'rotation_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
