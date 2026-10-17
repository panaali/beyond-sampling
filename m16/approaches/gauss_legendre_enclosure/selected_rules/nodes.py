"""Exactly four frozen new rules, one durable bounded process per order."""
from pathlib import Path
from fractions import Fraction as F
import sys,json,hashlib,time
import numpy as np
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent))
from exact_extended import Arithmetic,isolate,weight_interval,moments,signs_and_variations,pack,ROOT_BITS,WEIGHT_BITS
from run import distance

def main(n):
    out=H/f'nodes_{n}.json';progress=H/f'nodes_{n}_progress.jsonl';assert not out.exists() and not progress.exists()
    P=json.loads((H/'nodes_protocol.json').read_text())
    # Entire snapshot is byte-identical after reversing ONLY accepted-order guard.
    snapshot=(H/'exact_extended.py').read_text();original=(H.parent/'exact.py').read_text()
    assert snapshot.replace('assert n in [2,6,10,12,16,22,28,30] and bits>0','assert n in [2,6,10,16] and bits>0')==original
    assert n in P['orders']==[12,22,28,30] and ROOT_BITS==160 and WEIGHT_BITS==128
    for p,h in P['pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    controls=json.loads(Path(P['controls']).read_text());assert controls['cases']==13 and not controls['failures']
    def record(r):
        with progress.open('a') as f:f.write(json.dumps(pack(r),allow_nan=False)+'\n')
    start=time.process_time();record(dict(stage='order_attempt',n=n));A=Arithmetic();cert=isolate(n,A);positive=cert['positive_roots'];weights=[];details=[]
    for lo,hi in positive:
        ls,lv=signs_and_variations(n,lo,A);hs,hv=signs_and_variations(n,hi,A)
        assert ls[-1] and hs[-1] and lv-hv==1 and hi-lo==F(1,1<<ROOT_BITS)
        w=weight_interval(n,(lo,hi),A);weights.append(w)
        details.append(dict(root=(lo,hi),left_signs=ls,right_signs=hs,left_variations=lv,right_variations=hv,weight=w))
    bounded=not cert['unresolved'] and len(positive)==n//2 and all(w['status']=='bounded' for w in weights)
    checks=moments(n,positive,[w['dyadic'] for w in weights],A) if bounded else None
    if checks is not None:assert all(r['contains'] for r in checks)
    record(dict(stage='exact_certificate_complete',n=n,bounded=bounded,details=details,unresolved=cert['unresolved']))
    record(dict(stage='source_constants_attempt',n=n))
    t=time.process_time();x,w=np.polynomial.legendre.leggauss(n);source_cpu=time.process_time()-t
    assert x.shape==w.shape==(n,) and np.isfinite(x).all() and np.isfinite(w).all()
    comparison=[]
    if bounded:
        roots=[(-hi,-lo) for lo,hi in reversed(positive)]+positive
        ws=[r['dyadic'] for r in reversed(weights)]+[r['dyadic'] for r in weights]
        for j,(root,weight) in enumerate(zip(roots,ws)):
            comparison.append(dict(index=j,ideal_node=root,ideal_weight=weight,node=distance(x[j],root,A),weight=distance(w[j],weight,A)))
    row=dict(n=n,bounded=bounded,positive_certificates=details,unresolved=cert['unresolved'],visited_intervals=cert['visited'],moments=checks,source_comparison=comparison,source_nodes_hex=[float(v).hex() for v in x],source_weights_hex=[float(v).hex() for v in w],counts=dict(A.count),peak_bits=A.peak,cpu_seconds=time.process_time()-start,source_leggauss_cpu_seconds=source_cpu)
    record(dict(stage='order_complete',**row))
    result=dict(complete=True,order=row,source_leggauss_calls=1,PRECOMPUTE_FLOPS=None,precompute_status='unmeasured native/hardware work; exact universal setup categories retained; no ranked entry',numpy=np.__version__,python=sys.version,scope=P['scope'],model_mass_geometry_tensor_MC_calls=0)
    out.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');print(json.dumps(dict(n=n,bounded=bounded,cpu_seconds=row['cpu_seconds'])))
if __name__=='__main__':
    try:main(int(sys.argv[1]))
    except BaseException as e:
        with (H/'failures.jsonl').open('a') as f:f.write(json.dumps(dict(argument=sys.argv[1:],type=type(e).__name__,message=str(e)))+'\n')
        raise
