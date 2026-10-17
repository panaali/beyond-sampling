"""Complete existing T95 panel, fixed chunks and untouched variant definitions."""
import time
START=time.process_time()
import argparse,hashlib,json,resource
from pathlib import Path
import numpy as np
import torch
import adapter as A
from exact import exact_acc
from m43 import Ops as ExactOps

HERE=Path(__file__).resolve().parent


def models(h):
    manifest=json.loads((HERE/'panel_checkpoint_manifest.json').read_text());base={}
    for cp in manifest['rows']:
        if cp['H']!=h:continue
        path=Path(cp['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==cp['sha256']
        d=torch.load(path,weights_only=True,map_location='cpu');m=tuple(d[k].numpy().astype(float) for k in ['rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'])
        base[cp['seed']]=(m[0][:,0],m[1],m[2])
    out={f'seed{s}':base[s] for s in range(5)};m=base[{4:0,5:1,6:3}[h]]
    for sigma in [.0025,.005,.01,.02]:
        rng=np.random.default_rng(0)
        for k in range(6):out[f's{sigma}_{k}']=tuple(w+sigma*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape) for w in m)
    for ns in [0,1]:
        rng=np.random.default_rng(ns)
        for k in range(64):out[f'abs0.05_n{ns}_{k}']=tuple(w+.05*rng.standard_normal(w.shape) for w in m)
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--H',type=int,choices=[4,5,6],required=True);parser.add_argument('--start',type=int,required=True);parser.add_argument('--count',type=int,default=50);args=parser.parse_args()
    assert 0<=args.start<157 and 1<=args.count<=50
    reference=json.loads((HERE/'suite_reference.json').read_text())[str(args.H)];allmodels=models(args.H)
    assert list(allmodels)==list(reference['rows'])
    selected=list(allmodels.items())[args.start:args.start+args.count]
    dest=HERE/f'suite_h{args.H}_{args.start:03d}.json';assert not dest.exists(),'Refuse overwriting completed or partial evidence'
    out=dict(complete=False,H=args.H,start=args.start,count=len(selected),models=[],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reference_sha256=hashlib.sha256((HERE/'suite_reference.json').read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256((HERE/'adapter.py').read_bytes()).hexdigest())
    def save():
        out['cpu_seconds_including_imports']=time.process_time()-START
        dest.write_text(json.dumps(out,indent=2)+'\n')
    for name,model in selected:
        start=time.process_time();ops=ExactOps();truth=exact_acc(model,ops);previous=reference['rows'][name]
        row=dict(name=name,truth=truth,old_truth=previous['truth'],truth_difference=truth-previous['truth'],
            weight_sha256=hashlib.sha256(b''.join(w.tobytes() for w in model)).hexdigest(),
            validation=dict(cpu_seconds=time.process_time()-start,proxy_ops=ops.n),methods={})
        if abs(row['truth_difference'])>1e-10:
            row['failure']='Pinned benchmark truth mismatch';out['models'].append(row);save();raise AssertionError(row['failure'])
        for mode in ['reference','omit','capped']:
            start=time.process_time()
            try:r=A.estimate(model,mode)
            except Exception as exc:r=dict(error=repr(exc))
            r['cpu_seconds']=time.process_time()-start
            if 'estimate'in r:
                r['bias']=r['estimate']-truth;r['mse_ratio_proxy']=r['bias']**2*r['proxy_ops']/(3*args.H*(args.H+2)*truth*(1-truth))
            row['methods'][mode]=r
            if time.process_time()-START>95:
                out['models'].append(row);out['stop_reason']='CPU deadline';save();return
        if 'estimate'in row['methods']['reference']:row['reference_difference_from_prior']=row['methods']['reference']['estimate']-previous['reference']['est']
        out['models'].append(row);save()
    out.update(complete=True,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save()
    print(json.dumps(dict(H=args.H,start=args.start,count=len(selected),cpu=out['cpu_seconds_including_imports'],
        max_truth_difference=max(abs(r['truth_difference']) for r in out['models']),
        max_ratios={m:max(r['methods'][m]['mse_ratio_proxy'] for r in out['models']) for m in ['reference','omit','capped']})),flush=True)


if __name__=='__main__':main()
