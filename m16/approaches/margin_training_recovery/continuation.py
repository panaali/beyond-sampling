"""Frozen T115 missing-k30 continuation; original estimators/counters, independent pilot truth."""
from pathlib import Path
import sys,json,time,hashlib,importlib.util
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'dependencies'))
from m43 import Ops
from reference import exact
from zoo43 import noisy_copies
spec=importlib.util.spec_from_file_location('t115_pinned',HERE/'frozen/est75_t112.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
RECIPES={'r85b':P.RECIPE85B,'r112':P.RECIPE112}
RUNG=json.loads((HERE/'frozen/rung_k30.json').read_text())['seeds']
OLD=json.loads((HERE/'frozen/margin_k30.json').read_text())['seeds']['0']
MANIFEST=json.loads((HERE/'continuation_manifest.json').read_text())

def save(path,d):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(path)

def model_hash(m):return hashlib.sha256(b''.join(a.tobytes() for a in m)).hexdigest()

def load(seed):
    import torch
    record=MANIFEST[f'checkpoint_{seed}'];path=Path(record['source'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
    sd=torch.load(path,weights_only=True,map_location='cpu')
    return tuple(sd[k].double().numpy().copy() for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))[0][:,0],sd['rnn.weight_hh_l0'].double().numpy().copy(),sd['linear.weight'].double().numpy().copy()

def run(m):
    out={}
    for name,recipe in RECIPES.items():
        P.STATS.update(cut=0,resolved=0,cells=0);o=Ops()
        value=float(P.est75(m,o,**recipe))
        out[name]=dict(value=value,native_ops=o.n,cells=P.STATS['cells'])
    return out

def score(seed,index,m,independent):
    t=time.process_time();r=RUNG[str(seed)]
    truth=r['clean']['truth'] if index<0 else r['panel']['truths'][index]
    out=dict(seed=seed,index=index,model_hash=model_hash(m),saved_truth=truth,recipes=run(m))
    if independent:
        q,n=exact(m);out.update(independent_truth=q,truth_leaves=n,truth_difference=q-truth)
        assert abs(q-truth)<1e-11,out
    if seed==0:
        for name,z in out['recipes'].items():
            expected=OLD['clean'][name]['err'] if index<0 else OLD['p20'][name]['errs'][index]
            assert abs(z['value']-truth-expected)<1e-12,out
            if index<0:assert z['native_ops']==OLD['clean'][name]['ops'] and z['cells']==OLD['clean'][name]['cells']
    out['cpu_seconds']=time.process_time()-t
    return out

def pilot():
    path=HERE/'continuation_pilot.json';assert not path.exists(),'Never overwrite completed evidence'
    start=time.process_time();out=dict(protocol_claim='4b6e3a4d',rows=[],complete=False)
    for seed in range(5):
        m=load(seed);out['rows'].append(score(seed,-1,m,True));save(path,out)
        for ns in range(20,24):
            for j,x in enumerate(noisy_copies(m,1 if seed==0 else 4,.05,ns)):
                index=(ns-20)*256+j
                out['rows'].append(score(seed,index,x,True));save(path,out)
    out.update(complete=True,cpu_seconds=time.process_time()-start)
    save(path,out);print(json.dumps(dict(rows=len(out['rows']),cpu_seconds=out['cpu_seconds'],max_truth_diff=max(abs(r['truth_difference']) for r in out['rows']))))

def chunk(seed,ns):
    pilot_data=json.loads((HERE/'continuation_pilot.json').read_text());assert pilot_data['complete']
    assert seed in range(1,5) and ns in range(20,24)
    path=HERE/f'k30_s{seed}_noise{ns}.json';assert not path.exists(),'Never overwrite completed evidence'
    start=time.process_time();base=load(seed);saved={r['index']:r for r in pilot_data['rows'] if r['seed']==seed}
    out=dict(seed=seed,noise_seed=ns,sigma=.05,rows=[],complete=False)
    for j,m in enumerate(noisy_copies(base,256,.05,ns)):
        index=(ns-20)*256+j
        if index in saved:
            row=saved[index];assert model_hash(m)==row['model_hash'];out['rows'].append(dict(row,reused_pilot=True))
        else:out['rows'].append(score(seed,index,m,False))
        out['cpu_seconds']=time.process_time()-start;save(path,out)
    out['complete']=True;save(path,out);print(json.dumps(dict(seed=seed,noise_seed=ns,n=len(out['rows']),cpu_seconds=out['cpu_seconds'])))

if __name__=='__main__':
    for name,r in MANIFEST.items():
        if not name.startswith('checkpoint_'):assert hashlib.sha256((HERE/'dependencies'/name).read_bytes()).hexdigest()==r['sha256']
    if sys.argv[1]=='pilot':pilot()
    else:chunk(int(sys.argv[1]),int(sys.argv[2]))
