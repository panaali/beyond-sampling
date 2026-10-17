"""Pinned scale control: full tracing and canonical readout before LOOK2."""
import time
START=time.process_time()
import hashlib,json,resource
from pathlib import Path
import numpy as np
import adapter as A
from exact import exact_acc
from m43 import Ops as ExactOps

HERE=Path(__file__).resolve().parent


def full_trace(model):
    diag=dict(forward_calls=0,not_at_clamp=0)
    def checked(cx,st,theta,ops):
        result=A.ORIGINAL_FORWARD(cx,st,theta,ops);diag['forward_calls']+=1
        diag['not_at_clamp']+=int(result[2] is not cx.fcl)
        return result
    old=A.F.LOOK;A.F.LOOK=64;A.F.forward_trace=checked
    ops=A.F.Ops();info={};start=time.process_time()
    try:value=A.F.est_fast(model,ops,info)
    finally:A.F.LOOK=old;A.F.forward_trace=A.ORIGINAL_FORWARD
    return dict(estimate=value,proxy_ops=ops.n,info={k:v for k,v in info.items() if k!='err_by'},
        diagnostics=diag,cpu_seconds=time.process_time()-start)


def main():
    pilot=json.loads((HERE/'pilot.json').read_text());base=tuple(np.array(w) for w in pilot['models'][0]['weights']);rows=[]
    for scale in [1e-8,1.,1e8]:
        model=(base[0],base[1],base[2]*scale);ops=ExactOps();start=time.process_time();truth=exact_acc(model,ops)
        reference=dict(accuracy=truth,proxy_ops=ops.n,cpu_seconds=time.process_time()-start)
        full=full_trace(model)
        factor=float(np.max(abs(model[2])));canonical=(model[0],model[1],model[2]/factor if factor else model[2].copy())
        methods={}
        for mode in ['reference','omit','capped']:
            start=time.process_time();row=A.estimate(canonical,mode);row['cpu_seconds']=time.process_time()-start
            row['normalization_proxy_ops']=3*model[2].size;row['proxy_ops']+=row['normalization_proxy_ops'];methods[mode]=row
        rows.append(dict(scale=scale,numerical_truth=reference,full_look=full,canonical_factor=factor,canonical_readout=canonical[2].tolist(),methods=methods))
        if time.process_time()-START>18:raise TimeoutError('review deadline')
    out=dict(complete=True,rows=rows,cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_pilot_sha256=hashlib.sha256((HERE/'pilot.json').read_bytes()).hexdigest())
    (HERE/'scale_diagnosis.json').write_text(json.dumps(out,indent=2)+'\n')
    for r in rows:print(r['scale'],r['numerical_truth']['accuracy'],r['full_look'],{k:v['estimate'] for k,v in r['methods'].items()})
    print('CPU',out['cpu_seconds_including_imports'])


if __name__=='__main__':main()
