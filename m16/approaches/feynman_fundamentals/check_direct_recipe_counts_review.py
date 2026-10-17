import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from decimal import Decimal as D,localcontext
from fractions import Fraction as F
import hashlib,json,math,time
import numpy as np
import torch
P=Path('/workspace/work/recover-small_h2_910/review/recipe112_recovery');start=time.process_time();summary=json.loads((P/'direct_summary.json').read_text());envelope=json.loads((P/'identified_cost_envelope.json').read_text());weight_cache={};coverage=[];totalcpu=0
scriptsha=hashlib.sha256((P/'core_count.py').read_bytes()).hexdigest()
def weight(H,z):
 if (H,z) not in weight_cache:
  manifest=json.loads((P/f'counter_H{H}.json').read_text())['checkpoint_hashes'];paths=[Path(p) for p in manifest if p.endswith(f'_{z}.pth')];assert len(paths)==1;path=paths[0];assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest[str(path)]
  s=torch.load(path,map_location='cpu',weights_only=True);weight_cache[H,z]=(tuple(a.numpy().astype(float) for a in [s['rnn.weight_ih_l0'][:,0],s['rnn.weight_hh_l0'],s['linear.weight']]),manifest[str(path)])
 return weight_cache[H,z]
def read(H,z,noise,border=False):
 global totalcpu
 name='borderline' if border else f'core_H{H}_z{z}_n{noise}';d=json.loads((P/'direct_counts'/f'{name}.json').read_text());assert d['complete'] and d['all_parity_pass'] and d['script_sha256']==scriptsha
 path=P/'snapshot/out'/('fresh_H8_rel0.0025_90.json' if border else f'fresh_H{H}_abs0.05_80-81-82-83.json');assert hashlib.sha256(path.read_bytes()).hexdigest()==d['source_sha256'];raw=json.loads(path.read_text());n=32 if border else 64;off=raw['seeds'].index(noise)*n;old=raw['zoo'][str(z)]['rows'][1:][off:off+n];assert len(d['rows'])==len(old)==n
 base,sha=weight(H,z);assert sha==d['checkpoint_sha256'];rng=np.random.default_rng(noise)
 for i,(r,o) in enumerate(zip(d['rows'],old)):
  assert r['copy']==i and 'failure' not in r and r['truth']==o['truth'] and r['saved_estimate']==o['est'] and r['saved_ops']==o['ops'];assert r['saved_cells']==o['cells']==r['stats']['cells'];assert r['parity_pass'] and r['cell_count_equal'] and abs(r['estimate']-o['est'])<=1e-12
  model=tuple(w+(.0025*math.sqrt(float((w*w).mean())) if border else .05)*rng.standard_normal(w.shape) for w in base);assert hashlib.sha256(b''.join(w.tobytes() for w in model)).hexdigest()==r['model_sha256']
 totalcpu+=d['cpu'];coverage.append(name);return d['rows'],d['sampler_ops']
def check(rows,sampler,saved):
 with localcontext() as ctx:
  ctx.prec=65;n=len(rows);err=[D(r['estimate'])-D(r['truth']) for r in rows];var=[D(r['truth'])*(1-D(r['truth'])) for r in rows];ops=[r['ops'] for r in rows];samples=[o//sampler for o in ops]
  mse=sum(e*e for e in err)/n;den=sum(v/k for v,k in zip(var,samples))/n;cont=sum(v*sampler/o for v,o in zip(var,ops))/n;meanbudget=sum(var)/n*sampler/(sum(ops)/D(n));vals={'mse':mse,'mean_sampling_variance_integer':den,'ratio_integer_budget':mse/den,'ratio_continuous_budget':mse/cont,'ratio_mean_budget':mse/meanbudget}
  for key,value in vals.items():assert math.isclose(float(value),saved[key],rel_tol=5e-15,abs_tol=1e-25),(key,float(value),saved[key])
  e2=np.array([float(e*e) for e in err]);vari=np.array([float(v/k) for v,k in zip(var,samples)]);ix=np.random.default_rng(0).integers(0,n,(1000,n));ci=np.quantile(e2[ix].sum(1)/vari[ix].sum(1),[.025,.975]);assert np.allclose(ci,saved['bootstrap95_integer'],rtol=5e-15,atol=1e-22)
  assert min(samples)==saved['min_samples'] and max(samples)==saved['max_samples'] and saved['cell_count_mismatches']==0 and saved['all_parity_pass'];return float(mse/den)
ratios={};envelopes={}
for H in (4,5,6,8):
 ratios[H]=[];envelopes[H]=[]
 for z in range(5):
  rows=[]
  for noise in (80,81,82,83):r,sampler=read(H,z,noise);rows+=r
  s=next(g for g in summary['groups'] if (g['H'],g['zoo'])==(H,z));ratios[H].append(check(rows,sampler,s));e=next(g for g in envelope['groups'] if (g['H'],g['zoo'])==(H,z));factor=1+max(F(1,3),F(4*H,3*H+8));assert F(e['factor'])==factor
  filled=[dict(r,ops=math.ceil(factor*r['ops'])+1) for r in rows];check(filled,sampler,e['identified_gap_envelope']);envelopes[H].append(check(filled,sampler-3,e['numeric_only_sampler_envelope']))
rows,sampler=read(8,2,90,True);border=check(rows,sampler,summary['borderline']);factor=F(2);filled=[dict(r,ops=2*r['ops']+1) for r in rows];check(filled,sampler,envelope['borderline']['identified_gap_envelope']);check(filled,sampler-3,envelope['borderline']['numeric_only_sampler_envelope'])
assert len(coverage)==81 and len(set(coverage))==81 and math.isclose(totalcpu,summary['cpu_replay'],rel_tol=1e-14)
out=dict(core_rows=5120,borderline_rows=32,shards=81,all_model_byte_hashes_reconstructed=True,worst_raw_ratios={h:max(v) for h,v in ratios.items()},worst_identified_numeric_envelope_ratios={h:max(v) for h,v in envelopes.items()},borderline_ratio=border,author_replay_cpu=totalcpu,review_cpu=time.process_time()-start,scope='Saved direct counts/model-weight reconstruction and conditional fixed-panel bootstrap only; no forward/estimator rerun. Identified-gap envelope is not full native cost; original truth, tie and sampler-category qualifications persist.')
Path(__file__).with_name('direct_recipe_counts_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
