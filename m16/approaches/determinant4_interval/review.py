"""Frozen exact-rational determinant comparison; no Gaussian mass evaluation."""
import time
START=time.process_time()
from fractions import Fraction as F
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import resource
import sys
import numpy as np
from primitive import Counts,determinant47,point_matrix,interval_matrix

HERE=Path(__file__).resolve().parent
BASE=Path('/workspace/work/recover-small_h2_34/m16/approaches')
DATA=BASE/'class_region_t4/quadratic_exact_mass_calls.npz'
OLD=BASE/'positive_mass_budget/primitive.py'
EXPECTED={DATA:'07a0720e229bd2e0aeda511b60038efdc37f76c17aadfdeb17cf0caf3f264581',OLD:'0397ce32b78968c603f9d90faee276a790851b92b5c9600d6ae1f89b86780107'}
FIXED16=(0,2115,4231,6346,3222,3219,3221,3224,5962,3251,3254,3836,2567,2577,5448,2020)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source():
    for p,h in EXPECTED.items():assert sha(p)==h,str(p)
    spec=importlib.util.spec_from_file_location('pinned_old_positive',OLD)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def exact_det(A):
    """Independent exact elimination with row swaps, not a minor formula."""
    B=[[F(float(v)) for v in row] for row in A];answer=F(1)
    for j in range(4):
        pivot=next((i for i in range(j,4) if B[i][j]!=0),None)
        if pivot is None:return F(0)
        if pivot!=j:B[j],B[pivot]=B[pivot],B[j];answer=-answer
        v=B[j][j];answer*=v
        for i in range(j+1,4):
            q=B[i][j]/v
            for k in range(j+1,4):B[i][k]-=q*B[j][k]
    return answer

def evaluate(A,old):
    exact=exact_det(A);point_rows={}
    for name,fn,target in [('new47',determinant47,47),('old95',old.determinant,95)]:
        count=Counts();value=fn(point_matrix(A,count)).value
        assert count.arithmetic==target and count.comparisons==0
        if math.isfinite(value):
            error=F(value)-exact
            row=dict(value=value,absolute_error=float(abs(error)),relative_error=float(error/exact) if exact else None,status='finite')
        else:row=dict(value=None,absolute_error=None,relative_error=None,status='nonfinite')
        row['counts']=count.dictionary();point_rows[name]=row
    count=Counts()
    try:
        interval=determinant47(interval_matrix(A,count));lo,hi=interval.lo,interval.hi
        enclosed=F(lo)<=exact<=F(hi)
        # Status/summary bookkeeping is outside the primitive operation count.
        status='positive' if lo>0 else 'negative' if hi<0 else 'ambiguous'
        width=F(hi)-F(lo)
        bounds=dict(status=status,lo=lo,hi=hi,contains_exact=enclosed,
                    relative_width=float(width/abs(exact)) if exact else None,counts=count.dictionary())
    except ArithmeticError as exc:
        bounds=dict(status='unsupported',reason=str(exc),contains_exact=None,counts=count.dictionary())
    return dict(exact_determinant=str(exact),exact_sign=int(exact>0)-int(exact<0),point=point_rows,interval=bounds)

def save(name,out):
    out.update(cpu_seconds_including_imports=time.process_time()-START,
               max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               source_hashes={str(p):sha(p) for p in (*EXPECTED,HERE/'primitive.py',Path(__file__))})
    (HERE/name).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')

def controls():
    old=source();eye=np.eye(4);cases=[('zero',np.zeros((4,4)),F(0))]
    for permutation in itertools.permutations(range(4)):
        sign=(-1)**sum(permutation[i]>permutation[j] for i in range(4) for j in range(i+1,4))
        cases.append(('permutation_'+''.join(map(str,permutation)),eye[list(permutation)],F(sign)))
    duplicate=eye.copy();duplicate[3]=duplicate[2];cases.append(('duplicate',duplicate,F(0)))
    triangular=np.array([[1,.5,.25,.125],[0,2,.25,.125],[0,0,3,.5],[0,0,0,4]],float)
    cases.append(('dyadic_triangular',triangular,F(24)))
    H=np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],float)/2
    for k in (10,30,45,52):
        A=H.copy();A[3]*=2.**-k;cases.append((f'hadamard_thin_{k}',A,F(1,2**k)))
    for k in (300,-300):
        cases.append((f'diagonal_2^{k}',eye*2.**k,F(2)**(4*k)))
    rows=[]
    for name,A,expected in cases:
        row=evaluate(A,old);assert F(row['exact_determinant'])==expected
        assert row['interval']['contains_exact'] is not False
        row.update(name=name,expected_determinant=str(expected));rows.append(row)
    # Validate subnormal preservation and one-step outward arithmetic used above.
    tiny=math.nextafter(0.,1.);assert tiny>0 and F(tiny)==F(1,2**1074)
    c=Counts();a=interval_matrix([[tiny]*4]*4,c)[0][0];prod=a*a
    assert F(prod.lo)<=F(tiny)**2<=F(prod.hi)
    out=dict(complete=True,controls=rows,ieee_subnormal_control=True)
    save('controls.json',out)
    print(json.dumps(dict(controls=len(rows),unsupported=[r['name'] for r in rows if r['interval']['status']=='unsupported'],
                         ambiguous=[r['name'] for r in rows if r['interval']['status']=='ambiguous'],cpu=out['cpu_seconds_including_imports'])))

def corpus(chunk):
    old=source();rays=np.load(DATA,allow_pickle=False)['rays'];assert rays.shape==(6347,4,4)
    start=1600*chunk;end=min(len(rays),start+1600);assert 0<=chunk<4
    out=dict(complete=False,chunk=chunk,begin=start,end=end,rows=[])
    for i in range(start,end):
        if time.process_time()-START>18:
            out['stop_reason']='18CPU soft cap';save(f'corpus_{chunk}.json',out);return
        row=evaluate(rays[i],old);row.update(index=i,prior_fixed16=i in FIXED16);out['rows'].append(row)
        if (i-start+1)%400==0:save(f'corpus_{chunk}.json',out)
    out['complete']=True;save(f'corpus_{chunk}.json',out)
    print(json.dumps(dict(chunk=chunk,cases=len(out['rows']),interval_failures=sum(r['interval']['contains_exact'] is False for r in out['rows']),
                         unresolved=sum(r['interval']['status'] in ('unsupported','ambiguous') for r in out['rows']),cpu=out['cpu_seconds_including_imports'])))

if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_CPU,(20,22))
    if sys.argv[1]=='controls':controls()
    else:corpus(int(sys.argv[2]))
