"""Frozen 30-model original/first-noise T99 discriminator, no recipe selection."""
import time
START=time.process_time()
import hashlib,json,resource
from pathlib import Path
import numpy as np
import torch
import adapter as A
from exact import exact_acc
from m43 import Ops as ExactOps
from scale_diagnosis import full_trace

HERE=Path(__file__).resolve().parent


def main():
    manifest=json.loads((HERE/'panel_checkpoint_manifest.json').read_text())
    out=dict(complete=False,models=[],source_pins=json.loads((HERE/'source_manifest.json').read_text()),
        checkpoint_manifest_sha256=hashlib.sha256((HERE/'panel_checkpoint_manifest.json').read_bytes()).hexdigest())
    dest=HERE/'panel.json'
    def save():
        out['cpu_seconds_including_imports']=time.process_time()-START
        dest.write_text(json.dumps(out,indent=2)+'\n')
    for cp in manifest['rows']:
        if 'error'in cp:raise RuntimeError(cp)
        path=Path(cp['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==cp['sha256']
        d=torch.load(path,weights_only=True,map_location='cpu')
        base=tuple(d[k].numpy().astype(float) for k in ['rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight']);base=(base[0][:,0],base[1],base[2])
        rng=np.random.default_rng(0);noise=tuple(w+.02*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape) for w in base)
        for kind,model in [('original',base),('relative02_first',noise)]:
            start=time.process_time();ops=ExactOps();truth=exact_acc(model,ops)
            row=dict(H=cp['H'],seed=cp['seed'],kind=kind,truth=truth,
                validation=dict(proxy_ops=ops.n,cpu_seconds=time.process_time()-start,
                    cost_note='Early reference map counters contain H4 literals; incomplete for H5/H6.'),methods={})
            for mode in ['reference','deferred','omit','capped','full']:
                start=time.process_time()
                try:r=full_trace(model) if mode=='full' else A.estimate(model,mode)
                except Exception as exc:r=dict(error=repr(exc))
                r['cpu_seconds']=time.process_time()-start
                if 'estimate'in r:
                    r['bias']=r['estimate']-truth;r['mse_ratio_proxy']=r['bias']**2*r['proxy_ops']/(3*cp['H']*(cp['H']+2)*truth*(1-truth))
                row['methods'][mode]=r
                if time.process_time()-START>95:
                    out['models'].append(row);out['stop_reason']='CPU deadline';save();return
            a=row['methods'];row['deferred_parity']=('estimate'in a['reference'] and 'estimate'in a['deferred'] and abs(a['reference']['estimate']-a['deferred']['estimate'])<1e-12)
            out['models'].append(row);save()
            print(cp['H'],cp['seed'],kind,{k:(v.get('bias'),v.get('proxy_ops')) for k,v in row['methods'].items()},flush=True)
    out.update(complete=True,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),adapter_sha256=hashlib.sha256((HERE/'adapter.py').read_bytes()).hexdigest())
    save()
    print('CPU',out['cpu_seconds_including_imports'],flush=True)


if __name__=='__main__':main()
