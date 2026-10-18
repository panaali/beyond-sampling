"""Independent complete captured-error replay aggregation, no integration rerun."""
from fractions import Fraction as F
from pathlib import Path
import json,math,time
import numpy as np
start=time.process_time();here=Path(__file__).parent;src=Path('/workspace/work/recover-small_h2_910/m16/approaches/leaf_mass_review');archive=np.load(src/'error_captured.npz');parents=archive['parent_ids'];capture=json.loads((src/'error_capture.json').read_text());assert capture['complete'] and capture['counts']['leaves']==8455 and capture['counts']['alloff']==capture['counts']['active_identical_readout_pairs']==0
seen=set();old=[];positive=[];bounds=[];cpu=[];neg=0;misses=0;rows=[]
for batch in range(8):
 d=json.loads((src/f'error_batch_{batch}.json').read_text());v=np.load(src/f'error_batch_{batch}.npz');assert d['complete'];ids=v['indices'].tolist();expected=np.flatnonzero((parents>=12*batch)&(parents<12*(batch+1))).tolist();assert ids==expected and not seen.intersection(ids);seen.update(ids)
 for key,field in [('source','source_sum'),('positive','positive_sum'),('bounds','conditional_integration_bound')]:assert math.fsum(v[key])==d[field]
 assert min(v['positive'])>=0 and min(v['bounds'])>=0;negative=int(sum(v['source']<0));assert negative==d['negative_source_count'];miss=int(sum(v['bounds']>d['per_simplex_bound_allowance']));assert miss==d['bound_target_misses'];assert not any(v['reused']);neg+=negative;misses+=miss;cpu.append(d['cumulative_cpu']);old.extend(v['source']);positive.extend(v['positive']);bounds.extend(v['bounds'])
 gate=np.load(src/f'full_batch_{batch}.npz')
 for j in range(12*batch,12*(batch+1),2):
  e=math.fsum(v['positive'][np.isin(parents[v['indices']],(j,j+1))]);g=math.fsum(gate['positive'][np.isin(np.load(src/'captured.npz')['parent_ids'][gate['indices']],(j,j+1))]);assert e<=g+1e-12
 rows.append(dict(batch=batch,simplices=len(ids),source_error_mass=d['source_sum'],positive_error_mass=d['positive_sum'],bound=d['conditional_integration_bound'],negative_count=negative,misses=miss))
assert seen==set(range(len(parents))) and len(seen)==35098
oldexact=sum(map(F,map(float,old)),F(0));newexact=sum(map(F,map(float,positive)),F(0));boundexact=sum(map(F,map(float,bounds)),F(0));oldacc=float(1-oldexact);newacc=float(1-newexact)
assert abs(oldacc-.9750473359398664)<2e-16
mc=[]
for name,n,p,rounding in [('torch_10M_reported',10000000,.97617,5e-6),('numpy_4M',4000000,.97606325,0.)]:
 se=math.sqrt(p*(1-p)/n);mc.append(dict(name=name,n=n,reported=p,report_rounding_radius=rounding,difference=newacc-p,nominal_z=(newacc-p)/se,scope='Existing sampling estimate; normal-SE descriptive, not exact truth or confidence certificate.'))
out=dict(batches=rows,captured_simplices=len(seen),source_accuracy=oldacc,positive_accuracy=newacc,error_mass_overstatement=float(oldexact-newexact),integration_only_bound=float(boundexact),negative_source_count=neg,bound_target_misses=misses,capture_cpu=capture['cpu_seconds_including_imports'],replay_cpu=math.fsum(cpu),existing_mc_comparisons=mc,exact_binary_source_error=str(oldexact),exact_binary_positive_error=str(newexact),review_cpu=time.process_time()-start,scope='Captured floating error geometry and saved numerical references; no complete floating certificate, cost win, or all-model ladder pass.')
(here/'error_mass_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('batches','exact_binary_source_error','exact_binary_positive_error')},indent=2))
