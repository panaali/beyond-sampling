"""Frozen H4T3 seed0/first-noise diagnostic, with analytic and parity controls."""
import time
START=time.process_time()
import hashlib,json,resource
from pathlib import Path
import numpy as np
import torch
import adapter as A
from exact import exact_acc
from m43 import Ops as ExactOps

HERE=Path(__file__).resolve().parent
MODES=['reference','deferred','omit','capped']


def run(model):
    out={}
    for mode in MODES:
        start=time.process_time()
        try:row=A.estimate(model,mode)
        except Exception as exc:row=dict(error=repr(exc))
        row['cpu_seconds']=time.process_time()-start;out[mode]=row
        if time.process_time()-START>95:raise TimeoutError('CPU deadline')
    if all('estimate' in x for x in out.values()):
        assert abs(out['reference']['estimate']-out['deferred']['estimate'])<1e-12
    return out


def main():
    out=dict(complete=False,controls=[],models=[],scales=[],source_pins=json.loads((HERE/'source_manifest.json').read_text()))
    path=HERE/'pilot.json'
    def save():path.write_text(json.dumps(out,indent=2)+'\n')
    # Separate constant label controls, including a nonzero hidden state with identical logits.
    for name,model in [
        ('zero_readout',(np.array([1.,-1.]),np.array([[.5,.2],[-.1,.3]]),np.zeros((3,2)))),
        ('duplicate_readout',(np.array([1.,-1.]),np.array([[.5,.2],[-.1,.3]]),np.ones((3,2)))),
        ('reset_constant_label',(np.array([1.,-1.]),np.zeros((2,2)),np.array([[1.,1.],[0.,0.],[-1.,-1.]])))]:
        rows=run(model);valid=all('estimate' in r and abs(r['estimate']-1/3)<1e-10 for r in rows.values())
        out['controls'].append(dict(name=name,truth=1/3,passed=valid,methods=rows));save()
    if not all(x['passed'] for x in out['controls']):
        out['stop_reason']='Analytic control failed; trained phase not run';save();print(out['stop_reason']);return
    cp=Path('/workspace/work/t99_checkpoints/H4_T3_seed0.pth')
    assert hashlib.sha256(cp.read_bytes()).hexdigest()==out['source_pins']['checkpoint_sha256']
    d=torch.load(cp,weights_only=True,map_location='cpu')
    model=tuple(d[k].numpy().astype(float) for k in ['rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight']);model=(model[0][:,0],model[1],model[2])
    rng=np.random.default_rng(0);noise=tuple(w+.02*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape) for w in model)
    for name,m in [('seed0',model),('seed0_relative02_first',noise)]:
        ops=ExactOps();stats={};start=time.process_time();truth=exact_acc(m,ops,stats);validation_cpu=time.process_time()-start
        rows=run(m)
        for row in rows.values():
            if 'estimate' in row:
                row['bias']=row['estimate']-truth
                row['mse_ratio_proxy']=(row['bias']**2)*row['proxy_ops']/(72*truth*(1-truth))
        out['models'].append(dict(name=name,weights=[w.tolist() for w in m],truth=truth,
            validation=dict(cpu_seconds=validation_cpu,proxy_ops=ops.n,**stats),methods=rows));save()
    for scale in [1e-8,1e8]:
        rows=run((model[0],model[1],scale*model[2]))
        out['scales'].append(dict(scale=scale,methods=rows));save()
    out.update(complete=True,cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),adapter_sha256=hashlib.sha256((HERE/'adapter.py').read_bytes()).hexdigest(),
        caveats='Inherited arithmetic proxies, model-independent setup not complete. Exact truth uses numerical spherical polygons. No rung pass claim.')
    save()
    print(json.dumps({k:v for k,v in out.items() if k not in ['controls','models','scales','source_pins']},indent=2))
    for row in out['models']:
        print(row['name'],row['truth'],{k:{x:y for x,y in v.items() if x in ['estimate','bias','proxy_ops','cpu_seconds','mse_ratio_proxy','diagnostics','info']} for k,v in row['methods'].items()})


if __name__=='__main__':main()
