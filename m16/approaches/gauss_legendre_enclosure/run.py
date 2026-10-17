"""Fixed node-only target certificates and source-binary component distances."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,time,sys
import numpy as np
from exact import Arithmetic,isolate,weight_interval,moments,signs_and_variations,pack,ROOT_BITS,WEIGHT_BITS
HERE=Path(__file__).resolve().parent

def distance(source,interval,A):
    x=F.from_float(float(source));lo,hi=interval
    a=A.op('sub',x,lo);b=A.op('sub',x,hi)
    A.count['rational_abs']+=2;A.count['extremum_comparisons']+=2
    upper=max(abs(a),abs(b))
    inside=A.compare('le',lo,x) and A.compare('le',x,hi)
    lower=F(0) if inside else min(abs(a),abs(b))
    return dict(source_hex=float(source).hex(),absolute_error_interval=(lower,upper),inside_ideal_enclosure=inside)

def main():
    out=HERE/'results.json';progress=HERE/'progress.jsonl'
    assert not out.exists() and not progress.exists()
    P=json.loads((HERE/'protocol.json').read_text());R=json.loads((HERE/'run_protocol.json').read_text())
    for name,pin in P['source_pins'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==pin
    for name,pin in R['pins'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==pin,name
    assert hashlib.sha256((HERE/'controls.json').read_bytes()).hexdigest()==R['completed_controls_sha256']
    c=json.loads((HERE/'controls.json').read_text());assert c['cases']==13 and c['failures']==[]
    assert P['orders']==[6,10,16] and P['root_bits']==ROOT_BITS and P['weight_bits']==WEIGHT_BITS
    def record(row):
        with progress.open('a') as f:f.write(json.dumps(pack(row),allow_nan=False)+'\n')
    rows=[];start=time.process_time()
    for n in P['orders']:
        record(dict(stage='order_attempt',n=n));A=Arithmetic();t=time.process_time()
        cert=isolate(n,A);positive=cert['positive_roots'];weights=[];details=[]
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
        source_t=time.process_time();x,w=np.polynomial.legendre.leggauss(n);source_cpu=time.process_time()-source_t
        assert x.shape==w.shape==(n,) and np.isfinite(x).all() and np.isfinite(w).all()
        comparison=[]
        if bounded:
            roots=[(-hi,-lo) for lo,hi in reversed(positive)]+positive
            ws=[r['dyadic'] for r in reversed(weights)]+[r['dyadic'] for r in weights]
            for j,(root,weight) in enumerate(zip(roots,ws)):
                comparison.append(dict(index=j,ideal_node=root,ideal_weight=weight,node=distance(x[j],root,A),weight=distance(w[j],weight,A)))
        row=dict(n=n,bounded=bounded,positive_certificates=details,unresolved=cert['unresolved'],visited_intervals=cert['visited'],moments=checks,source_comparison=comparison,source_nodes_hex=[float(v).hex() for v in x],source_weights_hex=[float(v).hex() for v in w],counts=dict(A.count),peak_bits=A.peak,cpu_seconds=time.process_time()-t,source_leggauss_cpu_seconds=source_cpu)
        rows.append(row);record(dict(stage='order_complete',**row))
    result=dict(orders=rows,complete=len(rows)==3,all_bounded=all(r['bounded'] for r in rows),source_leggauss_calls=3,source_native_work='unmeasured; source-generation entry calls retained, not priced as3 scalar operations',PRECOMPUTE_FLOPS=None,precompute_status='Model-free exact setup. Named rational/integer categories and peak bits retained; hardware FLOPs unmeasured; no ranked entry.',cpu_seconds=time.process_time()-start,python=sys.version,numpy=np.__version__,model_mass_geometry_tensor_MC_calls=0,scope=R['scope'])
    out.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(orders=[r['n'] for r in rows],all_bounded=result['all_bounded'],cpu_seconds=result['cpu_seconds'])))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        with (HERE/'failures.jsonl').open('a') as f:f.write(json.dumps(dict(phase='target',type=type(e).__name__,message=str(e)))+'\n')
        raise
