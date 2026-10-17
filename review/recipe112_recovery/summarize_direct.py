"""Direct per-case budget summaries; no model calls or transferred multipliers."""
import time
START=time.process_time()
import json,math
from fractions import Fraction as F
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;DATA=HERE/'direct_counts'
def mean(xs):
 xs=list(xs);return sum(xs,F(0))/len(xs)
def summarize(rows,sampler):
 errors=[F(r['estimate'])-F(r['truth']) for r in rows];v=[F(r['truth'])*(1-F(r['truth'])) for r in rows];ops=[r['ops'] for r in rows];n=[o//sampler for o in ops];assert min(n)>0
 squared=[e*e for e in errors];mse=mean(squared);den=mean(vv/F(nn) for vv,nn in zip(v,n));continuous=mean(vv*sampler/o for vv,o in zip(v,ops));mean_budget=mean(v)*sampler/mean(map(F,ops))
 e2=np.array(list(map(float,squared)));vari=np.array([float(vv/F(nn)) for vv,nn in zip(v,n)]);ix=np.random.default_rng(0).integers(0,len(rows),(1000,len(rows)));bs=e2[ix].mean(1)/vari[ix].mean(1)
 return dict(models=len(rows),mse=float(mse),mean_sampling_variance_integer=float(den),ratio_integer_budget=float(mse/den),ratio_continuous_budget=float(mse/continuous),ratio_mean_budget=float(mse/mean_budget),bootstrap95_integer=np.percentile(bs,[2.5,97.5]).tolist(),mean_ops=float(mean(map(F,ops))),sampler_ops=sampler,min_samples=min(n),max_samples=max(n),max_parity_difference=max(abs(r['parity_difference']) for r in rows),cell_count_mismatches=sum(not r['cell_count_equal'] for r in rows),all_parity_pass=all(r['parity_pass'] for r in rows))
def main(allow_partial=False):
 out=dict(groups=[],partial=allow_partial,files=[],cpu_replay=0.,scope='Finite fixed panel under declared scalar-reimplementation numeric units; conditional bootstrap only, no full native/hardware/generalization claim.')
 for H in (4,5,6,8):
  for zs in range(5):
   paths=[DATA/f'core_H{H}_z{zs}_n{n}.json' for n in (80,81,82,83)]
   if not all(p.exists() and json.loads(p.read_text())['complete'] for p in paths):
    assert allow_partial,'Missing complete shard';continue
   rows=[];samplers=set()
   for noise,p in zip((80,81,82,83),paths):
    d=json.loads(p.read_text());assert (d['H'],d['zoo'],d['noise'])==(H,zs,noise);assert len(d['rows'])==64 and [r['copy'] for r in d['rows']]==list(range(64));assert all('failure' not in r for r in d['rows'])
    rows+=d['rows'];samplers.add(d['sampler_ops']);out['cpu_replay']+=d['cpu'];out['files'].append(p.name)
   assert len(samplers)==1;g=summarize(rows,samplers.pop());g.update(H=H,zoo=zs);out['groups'].append(g)
 bp=DATA/'borderline.json'
 if bp.exists():
  b=json.loads(bp.read_text());assert b['complete'];out['borderline']=summarize(b['rows'],b['sampler_ops']);out['cpu_replay']+=b['cpu']
 out['rungs']=[dict(H=H,groups=sum(g['H']==H for g in out['groups']),worst_ratio=max(g['ratio_integer_budget'] for g in out['groups'] if g['H']==H),largest_ci_upper=max(g['bootstrap95_integer'][1] for g in out['groups'] if g['H']==H)) for H in (4,5,6,8) if any(g['H']==H for g in out['groups'])]
 out['cpu_summary']=time.process_time()-START;out['complete']=len(out['groups'])==20
 (HERE/('direct_summary_partial.json' if allow_partial else 'direct_summary.json')).write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k not in ('groups','files')}))
if __name__=='__main__':
 import sys
 main('--partial' in sys.argv)
