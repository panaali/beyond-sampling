"""Frozen saved-only T184 coverage/budget contraction; no evaluator imports."""
from pathlib import Path
from decimal import Decimal as D,localcontext
import hashlib,json,time
import numpy as np
HERE=Path(__file__).resolve().parent
NOISES=(80,81,82,83)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def avg(xs):return sum(xs,D(0))/D(len(xs))
def dec(x):return D.from_float(x) if isinstance(x,float) else D(x)
def group(rows,sampler):
 with localcontext() as ctx:
  ctx.prec=80
  e2=[(dec(r['estimate'])-dec(r['truth']))**2 for r in rows]
  v=[dec(r['truth'])*(1-dec(r['truth'])) for r in rows]
  costs=[r['named_numeric'] for r in rows];ns=[c//sampler for c in costs];assert min(ns)>0
  vari=[x/D(n) for x,n in zip(v,ns)]
  cont=[x*D(sampler)/D(c) for x,c in zip(v,costs)]
  mean_cost=avg(list(map(D,costs)));uniform_n=int(mean_cost//D(sampler));assert uniform_n>0
  uniform=avg(v)/D(uniform_n);uniform_cont=avg(v)*D(sampler)/mean_cost
  mse=avg(e2);den=avg(vari)
  A=np.array([float(x) for x in e2]);B=np.array([float(x) for x in vari]);V=np.array([float(x) for x in v]);C=np.array(costs,dtype=float)
  ix=np.random.default_rng(0).integers(0,len(rows),(1000,len(rows)))
  numer=A[ix].mean(1);matched=numer/B[ix].mean(1)
  # Resample cost with the same rows; uniform floor is recomputed each replicate.
  un=np.floor(C[ix].mean(1)/sampler);ubs=numer/(V[ix].mean(1)/un)
  ratios=[x/y for x,y in zip(e2,vari)]
  def textd(x):return str(x)
  return {'models':len(rows),'sampler_numeric':sampler,'mse':textd(mse),'mean_numeric':textd(mean_cost),
   'mean_legacy_proxy':textd(avg([D(r['legacy']) for r in rows])),
   'mean_sampling_variance_per_model_floor':textd(den),'ratio_per_model_floor':textd(mse/den),
   'ratio_per_model_continuous':textd(mse/avg(cont)),
   'uniform_floor_samples':uniform_n,'uniform_floor_total_samples':uniform_n*len(rows),
   'per_model_total_samples':sum(ns),'ratio_uniform_floor':textd(mse/uniform),
   'ratio_uniform_continuous':textd(mse/uniform_cont),
   'bootstrap95_per_model_floor':np.percentile(matched,[2.5,97.5]).tolist(),
   'bootstrap95_uniform_floor':np.percentile(ubs,[2.5,97.5]).tolist(),
   'individual_losses':sum(x>1 for x in ratios),'individual_ties':sum(x==1 for x in ratios),
   'max_individual_ratio':textd(max(ratios)),'min_samples':min(ns),'max_samples':max(ns)}

def main():
 start=time.process_time();out=HERE/'summary.json';assert not out.exists()
 gate=json.loads((HERE/'summary_gate.json').read_text())
 for p,s in gate['pins'].items():assert sha(Path(p))==s,p
 panelgate=sha(HERE/'panel_gate.json');paths=[];groups=[];blocks=[];allrows=[];losses=[]
 total_cpu=D(0);calls=0;reuse=0;numeric=0;integer=0
 for h in (4,5,6):
  for z in range(5):
   selected=[]
   for noise in NOISES:
    path=HERE/'panel'/f'H{h}_z{z}_n{noise}.json';d=json.loads(path.read_text())
    assert d['complete'] and (d['H'],d['zoo'],d['noise'])==(h,z,noise) and d['gate_sha256']==panelgate
    assert d['sampler_numeric']==4*h*h+12*h-1 and [r['copy'] for r in d['rows']]==list(range(64))
    archive=path.with_suffix('.jsonl.gz');assert sha(archive)==d['archive_sha256']
    original=HERE/'frozen/core'/f'core_H{h}_z{z}_n{noise}.json';assert sha(original)==d['prior_shard_sha256']
    old=json.loads(original.read_text());assert len(old['rows'])==64
    for r,s in zip(d['rows'],old['rows'],strict=True):
     assert r['stage']=='complete' and r['parity'] and r['hex']==float(r['estimate']).hex()==float(s['saved_estimate']).hex()
     assert (r['truth'],r['model_sha256'],r['legacy'],r['cells'])==(s['truth'],s['model_sha256'],s['saved_ops'],s['saved_cells'])
     assert r['reused_pilot']==(z==0 and noise==80 and r['copy']==0)
     r=dict(r,H=h,zoo=z,noise=noise);selected.append(r);allrows.append(r)
     with localcontext() as ctx:
      ctx.prec=80;p=dec(r['truth']);e=dec(r['estimate'])-p;n=r['named_numeric']//d['sampler_numeric'];ratio=e*e/(p*(1-p)/D(n))
      if ratio>=1:losses.append({'H':h,'zoo':z,'noise':noise,'copy':r['copy'],'model_sha256':r['model_sha256'],'ratio':str(ratio),'tie':ratio==1})
    b=group(d['rows'],d['sampler_numeric']);b.update(H=h,zoo=z,noise=noise);blocks.append(b)
    assert d['source_calls_started']==d['source_calls_completed']==2*(64-d['pilot_reuses'])
    total_cpu+=dec(d['cpu_seconds']);calls+=d['source_calls_completed'];reuse+=d['pilot_reuses']
    numeric+=sum(r['named_numeric'] for r in d['rows']);integer+=sum(r['integer_control'] for r in d['rows'])
    paths.append({'path':str(path),'sha256':sha(path),'archive_sha256':d['archive_sha256']})
   g=group(selected,4*h*h+12*h-1);g.update(H=h,zoo=z);groups.append(g)
 assert len(paths)==60 and len(allrows)==3840 and calls==7674 and reuse==3
 assert len({(r['H'],r['zoo'],r['noise'],r['copy']) for r in allrows})==3840
 assert sum(g['individual_losses']+g['individual_ties'] for g in groups)==len(losses)
 pilots=json.loads((HERE/'pilots.json').read_text());prov=json.loads((HERE/'provenance.json').read_text())
 result={'complete':True,'groups':groups,'noise_blocks':blocks,'individual_losses_or_ties':losses,'files':paths,
  'rows':3840,'source_calls_new_panel':calls,'source_calls_total_including_pilots':calls+6,'pilot_reuses':reuse,
  'named_numeric_total':numeric,'integer_control_total':integer,'panel_cpu_seconds':str(total_cpu),
  'provenance_pilots_panel_cpu_seconds':str(total_cpu+dec(pilots['cpu_seconds'])+dec(prov['cpu_seconds'])),
  'worst_group_ratio':str(max(D(g['ratio_per_model_floor']) for g in groups)),
  'largest_group_bootstrap95_upper':max(g['bootstrap95_per_model_floor'][1] for g in groups),
  'scope':'Fixed original noisy panel and saved numerical truths; named source arithmetic only. Conditional row bootstrap holds each saved truth/estimate/count fixed and resamples paired rows within each group. No truth uncertainty, hardware/native-library, model-free setup or generalization closure.',
  'allocation_note':'Per-model floors use each cost. Uniform floor uses floor(mean cost/sampler); continuous variants use unrounded budgets. Historical legacy proxy means are preserved without mixing units.',
  'summary_gate_sha256':sha(HERE/'summary_gate.json'),'cpu_seconds':time.process_time()-start}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('groups','noise_blocks','individual_losses_or_ties','files')}))
if __name__=='__main__':main()
