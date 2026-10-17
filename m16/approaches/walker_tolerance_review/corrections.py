"""Separately labeled v2 controls and minimal local legacy repairs."""
import hashlib
import json
import time
import types
import numpy as np
from scipy.special import ndtr
from check import HERE,ROOT,load,fixtures


def patched(label):
    path=ROOT/('sci_tao/core.py' if label=='tao' else 'tube/tube.py')
    source=path.read_text()
    if label=='tao':
        old='(np.abs(c) <= 1e-12 * (1 + np.abs(L).max(1))[:, None])'
        assert source.count(old)==1
        source=source.replace(old,'(c == 0)')
        old='    return mass, cost, cells'
        assert source.count(old)==1
        source=source.replace(old,"    if len(act):\n        raise RuntimeError('local adapter: incomplete line integral')\n"+old)
    else:
        old='        never = (zer & (D0 < 0)).any(1)'
        assert source.count(old)==1
        source=source.replace(old,'        never = (zer & ((D0 < 0) | ((D0 == 0) & (np.arange(L.shape[1])[None,:] < t[:,None])))).any(1)')
    module=types.ModuleType('local_'+label)
    exec(compile(source,'<local review adapter>','exec'),module.__dict__)
    return module,hashlib.sha256(source.encode()).hexdigest()


def main():
    start=time.process_time()
    python=load('v2python','sci_tao_v2/core.py')
    compiled=load('v2compiled','sci_tao_v2/compiled.py')
    tao,taohash=patched('tao'); tube,tubehash=patched('tube')
    modules={'v2_python':python,'v2_compiled':compiled,'local_tao_tie_and_completion':tao}
    paths=['sci_tao_v2/core.py','sci_tao_v2/compiled.py','sci_tao_v2/walker.cpp','sci_tao_v2/_walker.so']
    output=dict(rows=[],caps=[],source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths if (ROOT/p).exists()},
                local_adapter_source_sha256={'tao':taohash,'tube':tubehash})
    references=json.loads((HERE/'results.json').read_text())['rows']
    for name,model,y,a,b in fixtures():
        ref=next(r['reference']['mass'] for r in references if r['name']==name)
        results={}
        for label,module in modules.items():
            try:
                mass,cost,cells=module.walk_mass(model,y[None],np.array([a]),np.array([b]),np.array([.25]),'23')
                results[label]=dict(mass=float(mass[0]),error=float(mass[0])-ref,cost=float(cost[0]),cells=int(cells[0]))
            except (RuntimeError,ValueError,ArithmeticError) as exc:results[label]=dict(failure=repr(exc))
        output['rows'].append(dict(fixture=name,reference=ref,results=results))
    name,model,y,a,b=fixtures()[4]
    for cap in (0,1):
        for label,module in modules.items():
            try:
                result=module.walk_mass(model,y[None],np.array([a]),np.array([b]),np.array([.25]),'23',max_cells=cap)
                record=dict(returned_mass=float(result[0][0]))
            except (RuntimeError,ValueError,ArithmeticError) as exc:record=dict(explicit_failure=repr(exc))
            output['caps'].append(dict(label=label,cap=cap,**record))
    name,model,y,a,b=fixtures()[0]
    d=np.zeros((1,10));d[0,a]=1/np.sqrt(2);d[0,b]=-1/np.sqrt(2)
    W=lambda a,b:ndtr(b)-ndtr(a)
    output['local_tube']=[]
    for K in (0,1):
        err,rem,cost=tube.walk_side(model,y[None],d,np.array([0.]),np.array([.25]),np.array([1]),K,Wf=W)
        output['local_tube'].append(dict(K=K,error=float(err[0]),remainder=float(rem[0]),truth=float(ndtr(.25)-.5)))
    output['cpu_seconds']=time.process_time()-start
    (HERE/'corrections.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
