"""Global core panel coverage/resources and independent exact budgets over audited groups."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal
from collections import Counter
import json,hashlib,time,resource
import numpy as np
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
P=Path('/workspace/work/recover-small_h2_78/review/recipe112_core_native');O=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();avg=lambda xs:sum(xs,F())/len(xs)
def close(text,value):assert abs(F(Decimal(text))-value)<=F(1,10**70)*max(abs(value),F(1,10**100))
def group(rows,sampler,saved):
 ps=[F(r['truth']) for r in rows];errs=[(F(r['estimate'])-p)**2 for r,p in zip(rows,ps)];pq=[p*(1-p) for p in ps];costs=[r['named_numeric'] for r in rows];ns=[c//sampler for c in costs];var=[v/n for v,n in zip(pq,ns)];mse=avg(errs);mean=avg(list(map(F,costs)));uniform=int(mean//sampler);ratios=[e/v for e,v in zip(errs,var)]
 vals={'mse':mse,'mean_numeric':mean,'mean_legacy_proxy':avg([F(r['legacy']) for r in rows]),'mean_sampling_variance_per_model_floor':avg(var),'ratio_per_model_floor':mse/avg(var),'ratio_per_model_continuous':mse/avg([v*sampler/c for v,c in zip(pq,costs)]),'ratio_uniform_floor':mse/(avg(pq)/uniform),'ratio_uniform_continuous':mse/(avg(pq)*sampler/mean),'max_individual_ratio':max(ratios)}
 for k,v in vals.items():close(saved[k],v)
 for k,v in {'models':len(rows),'sampler_numeric':sampler,'uniform_floor_samples':uniform,'uniform_floor_total_samples':uniform*len(rows),'per_model_total_samples':sum(ns),'individual_losses':sum(x>1 for x in ratios),'individual_ties':sum(x==1 for x in ratios),'min_samples':min(ns),'max_samples':max(ns)}.items():assert saved[k]==v
 ix=np.random.default_rng(0).integers(0,len(rows),(1000,len(rows)));A=np.array(list(map(float,errs)));V=np.array(list(map(float,var)));PQ=np.array(list(map(float,pq)));C=np.array(costs,dtype=float);num=A[ix].mean(1)
 for key,v in [('bootstrap95_per_model_floor',num/V[ix].mean(1)),('bootstrap95_uniform_floor',num/(PQ[ix].mean(1)/np.floor(C[ix].mean(1)/sampler)))]:assert np.allclose(saved[key],np.percentile(v,[2.5,97.5]),rtol=2e-13,atol=0)
 return ratios
outpath=O/'core_native_complete_review_results.json';assert not outpath.exists();d=read(P/'summary_recovered.json');assert d['complete'];gate=read(P/'summary_recovery_gate.json');assert d['summary_gate_sha256']==sha(P/'summary_recovery_gate.json')
for p,h in gate['pins'].items():assert sha(p)==h
rec=read(P/'recovery_gate.json');pilots=read(P/'pilots.json');prov=read(P/'provenance.json');failed=read(P/'panel/H4_z2_n80.json');assert not failed['complete'] and failed['source_calls_started']==failed['source_calls_completed']==4
files=[];allrows=[];seen=set();calls=0;reuses=Counter();counts=Counter();cpu=F();topology=[];losses=[];reviewpins={}
for H in (4,5,6):
 for z in range(5):
  path=O/f'core_native_H{H}_z{z}_review_results.json';review=read(path);reviewpins[str(path)]=sha(path);assert review['complete'] and (review['H'],review['zoo'])==(H,z);selected=[];counts.update(review['categories']);calls+=review['new_calls'];reuses.update(review['reuses'])
  for f in review['files']:
   p=Path(f['path']);assert sha(p)==f['sha256'] and sha(p.with_suffix('.jsonl.gz'))==f['archive_sha256'];files.append(f);s=read(p);noise=s['noise'];assert s['complete'] and 0<=s['cpu_seconds']<=110;cpu+=F(s['cpu_seconds'])
   for r in s['rows']:
    assert r['model_sha256'] not in seen;seen.add(r['model_sha256']);selected.append(dict(r,H=H,zoo=z,noise=noise));allrows.append(selected[-1])
   b=next(b for b in d['noise_blocks'] if (b['H'],b['zoo'],b['noise'])==(H,z,noise));group(s['rows'],review['sampler_numeric'],b)
  g=next(g for g in d['groups'] if (g['H'],g['zoo'])==(H,z));ratios=group(selected,review['sampler_numeric'],g)
  assert float(sum((F(r['estimate'])-F(r['truth']))**2 for r in selected)/sum(F(r['truth'])*(1-F(r['truth']))/(r['named_numeric']//review['sampler_numeric']) for r in selected))==review['ratio_per_model_floor']
  for r,v in zip(selected,ratios):
   if v>=1:losses.append((r,v))
  for t in review['topology_differences']:
   r=next(r for r in selected if (r['noise'],r['copy'])==(t['noise'],t['copy']));topology.append(dict(H=H,zoo=z,noise=t['noise'],copy=t['copy'],model_sha256=r['model_sha256'],native=t['native'],scalar=t['scalar']))
assert len(d['groups'])==15 and len(d['noise_blocks'])==len(files)==60 and len(allrows)==len(seen)==d['rows']==3840 and files==d['files']
assert calls==d['new_calls_in_completed_shards']==7670 and reuses==Counter(pilot=3,failed_case=2) and d['source_calls_total_including_pilots']==7680 and d['panel_calls_total']==7674
assert topology==d['scalar_native_topology_differences'] and len(losses)==len(d['individual_losses_or_ties'])
for (r,v),s in zip(losses,d['individual_losses_or_ties']):
 for k in ('H','zoo','noise','copy','model_sha256'):assert r[k]==s[k]
 close(s['ratio'],v);assert s['tie']==(v==1)
integer=sum(v for k,v in counts.items() if k.startswith('integer_control'));numeric=sum(counts.values())-integer;assert (numeric,integer)==(d['named_numeric_total'],d['integer_control_total'])
# Author accumulated CPU Decimals outside localcontext at28 digits; retain rounding tolerance.
actual=cpu+F(failed['cpu_seconds'])+F(pilots['cpu_seconds'])+F(prov['cpu_seconds']);charged=cpu+failed['reservation_cpu']+F(pilots['cpu_seconds'])+F(prov['cpu_seconds'])
for k,v in [('completed_shard_cpu_seconds',cpu),('provenance_pilots_panel_measured_cpu_seconds',actual),('research_cap_charged_cpu_seconds',charged)]:assert abs(F(Decimal(d[k]))-v)<F(1,10**20)
assert charged<=3000 and failed['reservation_cpu']==d['failed_attempt_charged_reserve']==110
out={'complete':True,'models':3840,'groups':15,'noise_groups':60,'numeric':numeric,'integer':integer,'individual_losses_or_ties':len(losses),'topology_differences':len(topology),'source_calls':7680,'pilot_reuses':3,'failed_case_reuses':2,'measured_research_cpu':float(actual),'charged_research_cpu':float(charged),'group_review_pins':reviewpins,'summary_sha256':sha(P/'summary_recovered.json'),'review_cpu':time.process_time()-start,'scope':'Reuses15 independent complete raw-site audits; exact own/uniform/continuous budgets, paired bootstrap, global identities/reuse/topology/resources checked. No estimator/truth replay, full native/runtime or universal performance claim.'}
outpath.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='group_review_pins'}))
