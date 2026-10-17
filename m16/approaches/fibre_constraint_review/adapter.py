"""Narrow, pinned-source transformation for T99; author snapshots untouched."""
import hashlib,inspect,json,math,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
for row in json.loads((HERE/'source_manifest.json').read_text())['files']:
    assert hashlib.sha256((HERE/row['local']).read_bytes()).hexdigest()==row['sha256']
sys.path.insert(0,str(HERE/'_source'))
import fibre_fast as F

ORIGINAL_FORWARD=F.forward_trace
ORIGINAL_INTEGRATE=F._integrate
MODE='reference'
DIAG={}


def finish_order(pending,piece_ok,cons,ops):
    if pending is None:return
    ops.add(1)
    if MODE!='deferred' and piece_ok:
        DIAG['dropped_calls']+=1;DIAG['dropped_pairs']+=len(pending[1])
    else:F._add_gate_cons(cons,*pending,ops)


source=inspect.getsource(ORIGINAL_FORWARD)
assert source.count('    for _ in range(64):')==1
source=source.replace('    for _ in range(64):','    for _ in range(64):\n        pending = None')
old='            _add_gate_cons(cons, st, ahead[ahead != kk], fk, rk, ops)'
assert source.count(old)==1
source=source.replace(old,'            pending = (st, ahead[ahead != kk], fk, rk)')
old='        n_ok = n_ok + 1 if piece_ok else 0'
assert source.count(old)==1
source=source.replace(old,'        finish_order(pending, piece_ok, cons, ops)\n'+old)
F.__dict__['finish_order']=finish_order
exec(compile(source,str(HERE/'_source/fibre_fast.py')+'[deferred-local]','exec'),F.__dict__)
DEFERRED_FORWARD=F.forward_trace
F.forward_trace=ORIGINAL_FORWARD


def capped_integrate(cx,st0,t0,t1,ops,stats):
    total=0.;stack=[(t0,t1)]
    while stack:
        a0,a1=stack.pop()
        if a1-a0<1e-13:continue
        tm=(a0+a1)/2;before=DIAG['dropped_calls']
        iv,cons=F.trace(cx,st0,tm,ops)
        a,b=cons.interval(tm,a0,a1,ops)
        if DIAG['dropped_calls']>before:
            new_a=max(a,tm-math.pi/128);new_b=min(b,tm+math.pi/128);ops.add(4)
            DIAG['capped_intervals']+=int(new_a>a or new_b<b)
            a,b=new_a,new_b
        stats['traces']+=1;stats['cons']+=sum(len(x) for x in cons.al)
        for lo,hi in iv:
            if hi is not None:total+=F._Sint(hi[0],hi[1],a,b);ops.add(12)
            if lo is not None:total-=F._Sint(lo[0],lo[1],a,b);ops.add(12)
        if a>a0:stack.append((a0,a))
        if b<a1:stack.append((b,a1))
    return total


def estimate(model,mode):
    global MODE,DIAG
    assert mode in ['reference','deferred','omit','capped']
    MODE=mode;DIAG=dict(dropped_calls=0,dropped_pairs=0,capped_intervals=0)
    F.forward_trace=ORIGINAL_FORWARD if mode=='reference' else DEFERRED_FORWARD
    F._integrate=capped_integrate if mode=='capped' else ORIGINAL_INTEGRATE
    ops=F.Ops();info={}
    try:value=F.est_fast(model,ops,info)
    finally:F.forward_trace=ORIGINAL_FORWARD;F._integrate=ORIGINAL_INTEGRATE
    return dict(estimate=value,proxy_ops=ops.n,info={k:v for k,v in info.items() if k!='err_by'},
        error_by_family={str(k):v for k,v in info['err_by'].items()},diagnostics=DIAG.copy())
