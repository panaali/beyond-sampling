"""Exact basic primal/dual certificates for first-index nonnegative readouts."""
import time
START=time.process_time()
import gzip,hashlib,itertools,json,resource
from fractions import Fraction as F
from pathlib import Path
P=Path(__file__).resolve().parent
SOURCE=Path('/workspace/clean-neurons/m16/approaches/small_ladder_review/rational_geometry/geometry_h4_t4.json.gz')
COUNT=dict(linear_solves=0,feasible_bases=0,singular_bases=0)
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def solve(M,b):
    COUNT['linear_solves']+=1;n=len(b);a=[list(map(F,row))+[F(v)] for row,v in zip(M,b)]
    for j in range(n):
        pivot=next((i for i in range(j,n) if a[i][j]),None)
        if pivot is None:COUNT['singular_bases']+=1;return None
        a[j],a[pivot]=a[pivot],a[j];z=a[j][j];a[j]=[v/z for v in a[j]]
        for i in range(n):
            if i!=j:
                z=a[i][j];a[i]=[x-z*y for x,y in zip(a[i],a[j])]
    return tuple(row[-1] for row in a)
def certificate(weights,label):
    T=len(weights);H=len(weights[0]);eq=(F(1),)*H+(F(0),);cons=[];names=[]
    for i in range(H):cons.append(tuple(F(q==i) for q in range(H))+(F(0),));names.append(f'h{i}_nonnegative')
    for other in range(T):
        if other==label:continue
        strict=other<label or label==0
        cons.append(tuple(a-b for a,b in zip(weights[label],weights[other]))+(F(-int(strict)),));names.append(f'beats_{other}'+('_margin' if strict else '_weak'))
    candidates=[]
    for active in itertools.combinations(range(len(cons)),H):
        M=[eq]+[cons[i] for i in active];z=solve(M,[1]+[0]*H)
        if z is None or any(dot(c,z)<0 for c in cons):continue
        COUNT['feasible_bases']+=1;candidates.append((z[-1],z,active,M))
    if not candidates:raise ArithmeticError('No feasible normalized readout candidate')
    optimum=max(row[0] for row in candidates);valid=None
    for value,z,active,M in candidates:
        if value!=optimum:continue
        dual=solve(list(zip(*M)),[0]*H+[1])
        if dual is not None and all(x<=0 for x in dual[1:]):valid=(z,active,dual);break
    if valid is None:raise ArithmeticError('No matching nonnegative dual')
    z,active,dual=valid;mu=[F(0)]*len(cons)
    for idx,value in zip(active,dual[1:]):mu[idx]=-value
    target=(F(0),)*H+(F(1),)
    assert tuple(dual[0]*eq[q]-sum(mu[i]*cons[i][q] for i in range(len(cons))) for q in range(H+1))==target
    assert dual[0]==optimum and all(v>=0 for v in mu)
    selected=max(range(T),key=lambda j:dot(weights[j],z[:-1]));reachable=(label==0 or optimum>0)
    if label>0 and reachable:assert selected==label
    return dict(label=label,optimum=str(optimum),optimum_float=float(optimum),nonzero_h_witness=list(map(str,z[:-1])),witness_first_argmax=selected,
        possible_in_nonnegative_superset=reachable,origin_selects=(label==0),dual_bound=str(dual[0]),dual_multipliers=list(map(str,mu)),constraint_names=names,
        notes='label0 origin always possible; its positive margin optimum is diagnostic only' if label==0 else 'positive earlier margin with weak later margins iff possible')
def run(weights):return [certificate(weights,j) for j in range(len(weights))]
def main():
    zero=[[F(0)]*4 for _ in range(4)];duplicate=[[F(i+1) for i in range(4)] for _ in range(4)];dominated=[[F(1)]*4,[F(0)]*4,[F(-1)]*4,[F(-2)]*4]
    controls=[]
    for name,w in [('zero',zero),('duplicate',duplicate),('dominated',dominated)]:
        rows=run(w);assert [r['possible_in_nonnegative_superset'] for r in rows]==[True,False,False,False];controls.append(dict(name=name,rows=rows))
    identity=[[F(i==j) for i in range(4)] for j in range(4)];rows=run(identity);assert all(r['possible_in_nonnegative_superset'] for r in rows);controls.append(dict(name='identity',rows=rows))
    stored=SOURCE.read_bytes();data=json.loads(gzip.decompress(stored));w=[[F(int(a),int(b)) for a,b in row] for row in data['model'][2]];rows=run(w)
    impossible=[r['label'] for r in rows if not r['possible_in_nonnegative_superset']]
    result=dict(complete=True,controls=controls,trained=rows,unreachable_labels=impossible,unconditional_error_floor=len(impossible)/len(w),checkpoint_sha256=data['checkpoint_sha256'],artifact_sha256=hashlib.sha256(stored).hexdigest(),counts=COUNT,cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    assert result['cpu_seconds_including_imports']<20
    (P/'readout.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['controls','trained']},indent=2));print([(r['label'],r['optimum_float']) for r in rows])
if __name__=='__main__':main()
