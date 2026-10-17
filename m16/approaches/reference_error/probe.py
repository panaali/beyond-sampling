"""Frozen T37 residual-shuffle ceiling; DEV diagnostics, not an estimator."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'m16'))
import leaderboard as lb


def reference(x):
    order=np.argsort(x,axis=1)
    a,b=order[:,-1],order[:,-2]
    xp=np.maximum(x,0)
    scores=xp+2*xp[np.arange(len(x)),a,None]
    scores[np.arange(len(x)),a]=xp[np.arange(len(x)),a]+2*xp[np.arange(len(x)),b]
    return scores,a,b


def centered(f):
    return f-f.mean(axis=1,keepdims=True)


def stats(actual,approx,ref,groups=None):
    if groups is None:
        groups=[np.arange(len(actual))]
    dd=[approx[j].astype(float)-actual[j] for j in groups]
    bias=float(np.mean([v.mean() for v in dd]))
    se=float(np.sqrt(sum(v.var(ddof=1)/len(v) for v in dd))/len(groups))
    status='advance' if abs(bias)+3*se<=.001 else 'rejected' if abs(bias)-3*se>.001 else 'inconclusive'
    return dict(bias=bias,paired_se=se,bias_minus_3se=bias-3*se,bias_plus_3se=bias+3*se,
        actual_accuracy=float(np.mean([actual[j].mean() for j in groups])),
        independent_accuracy=float(np.mean([approx[j].mean() for j in groups])),
        reference_accuracy=float(np.mean([ref[j].mean() for j in groups])),
        disagreement=float(np.mean([np.mean(approx[j]!=actual[j]) for j in groups])),
        n_comparisons=int(sum(map(len,groups))),min_cell_n=min(map(len,groups)),
        max_cell_n=max(map(len,groups)),independence_gate=status)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='/workspace/clean-neurons/m16/approaches/mechanism_checks/checkpoint.pth')
    args=p.parse_args(); started=time.process_time(); wall=time.monotonic()
    import torch
    path=Path(args.checkpoint); state=torch.load(path,map_location='cpu',weights_only=True)
    base=tuple(state[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
    base=(base[0][:,0],base[1],base[2])
    calibration=np.random.default_rng(37101).standard_normal((16384,10))
    train=np.random.default_rng(37102).standard_normal((131072,10))
    held=np.random.default_rng(37103).standard_normal((131072,10))
    gc,_,_=reference(calibration); gt,at,bt=reference(train); gh,ah,bh=reference(held)
    gc,gt,gh=map(centered,(gc,gt,gh))
    target=bh
    ref_ok=gh.argmax(axis=1)==target
    assert np.all(ref_ok[(held>0).sum(axis=1)>=2])
    global_perm=np.random.default_rng(37104).permutation(len(train))
    rng=np.random.default_rng(37105); trids=[]; heids=[]; groups=[]; offset=0
    for a in range(10):
        for b in range(10):
            if a==b: continue
            ti=rng.permutation(np.flatnonzero((at==a)&(bt==b)))
            hi=rng.permutation(np.flatnonzero((ah==a)&(bh==b)))
            n=min(len(ti),len(hi)); trids.extend(ti[:n]); heids.extend(hi[:n]);groups.append(np.arange(offset,offset+n));offset+=n
    trids=np.asarray(trids); heids=np.asarray(heids)
    scales={'fixed':(np.ones(len(train)),np.ones(len(held))),
            'radial':(np.linalg.norm(train,axis=1)/np.sqrt(10),np.linalg.norm(held,axis=1)/np.sqrt(10))}
    out=dict(protocol_commit='58778ab',checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        input_counts=dict(calibration=len(calibration),train=len(train),heldout=len(held)),
        input_seeds=[37101,37102,37103],pairing_seeds=[37104,37105],results=[],
        analytic_reference_accuracy=1-2**-10*(9/10+10*8/9),
        selected_position_pairs=len(groups),position_comparisons=len(trids),
        unused_rows_per_pool=len(train)-len(trids),model_forward_inputs=0)
    for sigma in (.0025,.02):
        model=lb._noisy(base,6,sigma,seed=1)[0]
        fc=centered(lb.forward(model,calibration))
        beta=float(np.sum(fc*gc)/np.sum(gc*gc)); assert beta>0
        ft=centered(lb.forward(model,train))/beta
        fh=lb.forward(model,held)
        actual=fh.argmax(axis=1)==target
        out['model_forward_inputs']+=len(calibration)+len(train)+len(held)
        for scale,(st,sh) in scales.items():
            eta=(ft-gt)/st[:,None]
            global_ok=(gh+sh[:,None]*eta[global_perm]).argmax(axis=1)==target
            pair_ok=(gh[heids]+sh[heids,None]*eta[trids]).argmax(axis=1)==target[heids]
            for law,stat in [('global',stats(actual,global_ok,ref_ok)),
                             ('positions',stats(actual[heids],pair_ok,ref_ok[heids],groups))]:
                row=dict(sigma=sigma,scale=scale,law=law,beta=beta,**stat)
                out['results'].append(row);print(json.dumps(row),flush=True)
    out.update(cpu_seconds=time.process_time()-started,wall_seconds=time.monotonic()-wall,
        shape_tests_performed=False,shape_followup_required=any(r['independence_gate']=='advance' for r in out['results']),
        uncertainty='Paired input/residual uncertainty conditional on independent calibration beta; no simultaneous coverage claim.',
        class_label='DEV diagnostic with fitted oracle residuals, not estimator')
    Path(__file__).with_name('results.json').write_text(json.dumps(out,indent=2)+'\n')
    if out['cpu_seconds']>300: raise RuntimeError('CPU cap exceeded')


if __name__=='__main__':main()
