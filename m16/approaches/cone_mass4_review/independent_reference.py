"""Positive simplex and separately derived Schur-complement MP reference."""
import json,time,resource,math
from pathlib import Path
import numpy as np
import mpmath as mp
from . import author_snapshot as M
from .check_controls import positive_mass
HERE=Path(__file__).parent

def mp_mass(rays,dps,start):
    calls=0
    with mp.workdps(dps):
        A=mp.matrix([[mp.mpf(float(x)) for x in row] for row in rays]);N=(A.T)**-1
        norms=[mp.sqrt(sum(N[i,k]**2 for k in range(4))) for i in range(4)]
        C=[[sum(N[i,k]*N[j,k] for k in range(4))/(norms[i]*norms[j]) for j in range(4)] for i in range(4)]
        pairs=[(i,j,*[k for k in range(4) if k not in (i,j)]) for i in range(4) for j in range(i+1,4)]
        def integrand(t):
            nonlocal calls
            calls+=1
            if time.process_time()-start>95:raise TimeoutError('Independent reference95CPU guard')
            terms=[]
            for i,j,k,l in pairs:
                r=t*C[i][j];a=t*C[i][k];b=t*C[j][k];c=t*C[i][l];d=t*C[j][l];e=t*C[k][l]
                D=1-r*r;A=D-a*a-b*b+2*r*a*b;B=D-c*c-d*d+2*r*c*d;P=e*D-a*c-b*d+r*(a*d+b*c)
                pc=P/mp.sqrt(A*B)
                if abs(pc)>1+mp.mpf(10)**(-dps+8):raise ArithmeticError('Reference partial correlation outside unit interval')
                pc=max(mp.mpf(-1),min(mp.mpf(1),pc));terms.append(C[i][j]*mp.asin(pc)/mp.sqrt(D))
            return sum(terms)
        val=mp.mpf(1)/16+sum(mp.asin(C[i][j]) for i,j,k,l in pairs)/(8*mp.pi)+mp.quad(integrand,[0,mp.mpf('.5'),mp.mpf('.9'),mp.mpf('.99'),mp.mpf('.9999'),1],maxdegree=8)/(4*mp.pi**2)
        return dict(value=float(val),decimal=mp.nstr(val,dps),precision=dps,callbacks=calls)

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(100,102));start=time.process_time();q=np.load(HERE.parent/'class_region_t4/quadratic_exact_mass_calls.npz');r,v=q['rays'],q['values']
    ids=list(dict.fromkeys([0,2115,4231,6346]+np.argsort(abs(np.linalg.det(r)))[:4].tolist()+np.argsort(v)[:4].tolist()+np.argsort(-v)[:4].tolist()))
    cases=[('quadratic',i,r[i]) for i in ids];other=np.load(HERE.parent/'class_region_t4/nearest_exact_mass_calls.npz')['rays']
    cases += [('quadratic',5826,r[5826])]+[('nearest',i,other[i]) for i in (420,495,10506)]
    out=dict(planned_cases=len(cases),original_subset=ids,rows=[],complete=False,reference_roundoff_certified=False)
    def save():
        out['cpu']=time.process_time()-start;(HERE/'independent_results.json').write_text(json.dumps(out,indent=2)+'\n')
    for kind,index,R in cases:
        row=dict(kind=kind,index=index);out['rows'].append(row);op=M.Ops()
        try:row['author']=float(M.mass4_rays(R,op))
        except Exception as exc:row['author_failure']=type(exc).__name__+': '+str(exc)
        row['author_ops']=op.n;t=time.process_time()
        try:
            if np.all(R.min(0)*R.max(0)>=0):
                value,bound=positive_mass(abs(R),20);row.update(reference_method='positive_simplex',reference=value,integration_only_bound=bound,nodes=8000)
            else:
                row['reference_method']='independent_mp_schur';row['precision_checks']=[]
                for dps in (50,70):
                    row['precision_checks'].append(mp_mass(R,dps,start));save()
                row['reference']=row['precision_checks'][-1]['value'];row['refinement_difference']=row['precision_checks'][-1]['value']-row['precision_checks'][0]['value']
            if 'author' in row:row['author_minus_reference']=row['author']-row['reference']
        except Exception as exc:row['reference_failure']=type(exc).__name__+': '+str(exc)
        row['reference_cpu']=time.process_time()-t;save();print(json.dumps({k:v for k,v in row.items() if k!='precision_checks'}),flush=True)
        if time.process_time()-start>95:break
    out['complete']=len(out['rows'])==len(cases);save()
if __name__=='__main__':main()
