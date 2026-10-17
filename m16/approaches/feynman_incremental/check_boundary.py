"""Frozen per-face analytic controls and small stochastic-baseline runs."""
import hashlib,json,time
from pathlib import Path
import mpmath as mp
import numpy as np
import torch
from .boundary_estimator import Engine
from .check_estimator import CHECKPOINT
from .adapter import WalkFailure
HERE=Path(__file__).parent

def main():
    start=time.process_time();mp.mp.dps=70;out=dict(primitive=[],cases=[],accuracy_scoring=False,method_is_sampling_baseline=True)
    zero=(np.array([1.,-1.]),np.array([[.2,.3],[-.1,.4]]),np.zeros((10,2)))
    engine=Engine(np.random.default_rng(20001))
    for kind in ('A','B'):
        G=engine.Z2.geom_A(zero,16,engine.rng,.15) if kind=='A' else engine.Z2.geom_B(zero,16,engine.rng,.15,.1)
        values,ledger=engine.lines(zero,G);a=G['d'].argmax(1);b=G['d'].argmin(1);truth=[]
        for ai,bi,low,high in zip(a,b,G['mlo'],np.maximum(np.minimum(G['tau'],G['mhi']),0)):
            mass=mp.mpf(0)
            if high>low:mass=(mp.erfc(mp.mpf(float(low))/mp.sqrt(2))-mp.erfc(mp.mpf(float(high))/mp.sqrt(2)))/2
            truth.append(float(mass)*G['w']*(int(ai!=0)+int(bi!=0)))
        difference=float(np.max(np.abs(values-truth)));assert difference<1e-12
        out['primitive'].append(dict(kind=kind,rows=16,maximum_absolute_difference=difference,known_line_flops=ledger['known_flops']))
    before=engine.counts.get('rejection_candidates',0);x,generated=engine.zone_draw(8)
    assert len(x)==8 and generated>=1000 and engine.counts['rejection_candidates']-before==generated
    out['rejection_control']=dict(requested=8,generated=generated,all_candidates_charged=True)
    try:engine.zone_draw(8,cap=1)
    except RuntimeError as exc:out['cap_control']=str(exc)
    else:raise AssertionError('Candidate cap ignored')
    out['primitive_diagnostic_ledger']=engine.snapshot(True)
    state=torch.load(CHECKPOINT,weights_only=True,map_location='cpu')
    model=(state['rnn.weight_ih_l0'].numpy()[:,0].astype(float),state['rnn.weight_hh_l0'].numpy().astype(float),state['linear.weight'].numpy().astype(float))
    out['checkpoint_sha256']=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    for name,seed,current in (('checkpoint_seed0',0,model),('checkpoint_seed1',1,model),('zero_readout_seed0',0,(model[0],model[1],np.zeros_like(model[2])))):
        if time.process_time()-start>100:raise TimeoutError('110CPU cycle guard')
        e=Engine(np.random.default_rng(seed))
        try:
            value,cost,ledger=e.estimate(current,512);assert ledger['known_flops']==sum(ledger['components'].values())
            row=dict(name=name,seed=seed,estimate=value,cost=cost,ledger=ledger)
            if 'zero' in name:row['analytic_truth']=.1
        except WalkFailure as exc:row=dict(name=name,seed=seed,failure=str(exc),ledger=exc.ledger)
        out['cases'].append(row);out['cpu_seconds']=time.process_time()-start
        (HERE/'boundary_results.json').write_text(json.dumps(out,indent=2)+'\n')
        print(json.dumps(dict(name=name,estimate=row.get('estimate'),failure=row.get('failure'),cpu=out['cpu_seconds'])),flush=True)
    out.update(complete=True,cpu_seconds=time.process_time()-start);(HERE/'boundary_results.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
