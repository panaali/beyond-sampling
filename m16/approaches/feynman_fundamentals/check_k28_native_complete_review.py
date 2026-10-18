"""Global coverage over five independently checked raw seeds plus clean controls."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import ast,json,hashlib,resource,time
import numpy as np
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
P=Path('/workspace/work/recover-small_h4_4/review/margin_k28_native');O=Path(__file__).resolve().parent;D=Path('/workspace/work/recover-small_h2_78/m16/approaches/margin_training_recovery/frozen')
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
outpath=O/'k28_native_complete_review_results.json';assert not outpath.exists()
d=read(P/'full_summary.json');assert d['complete'] and not d['historical_identity_verified']
for p,h in d['pins'].items():assert sha(p)==h,p
for name in ('pilot_gate.json','continuation_gate.json'):
 for p,h in read(P/name)['pins'].items():assert sha(p)==h,p
# Independent reviewer helper definitions only; no author modules or repeated seed checks.
tree=ast.parse((O/'check_k28_native_seed_review.py').read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('ledger','metrics')];exec(compile(tree,'reviewer helper definitions','exec'))
controls=read(P/'controls.json');recon=read(P/'reconstruction.json');margin=read(D/'margin_k28.json');rung=read(D/'rung_k28.json');names=('r85b','r112')
assert len(d['groups'])==5 and len(d['noise_groups'])==20 and len(d['individual'])==10240 and len(d['clean'])==5

def metric_equal(expected,actual):
 for conv,v in expected.items():
  for key,x in v.items():
   y=actual[conv][key]
   if isinstance(x,float):assert abs(x-y)<=2e-13*max(abs(x),abs(y),1e-300),(conv,key,x,y)
   elif key=='bootstrap95_integer':assert np.allclose(x,y,rtol=2e-13,atol=0)
   else:assert x==y,(conv,key,x,y)
files={};totals={n:Counter() for n in names};ctotals={n:Counter() for n in names};models=set();reuses=[];individual=[];cpu=controls['cpu'];maximum=controls['cpu'];rss=controls['rss_kib'];pairs=baselines=0;reviews={};clean_results=[]
for seed in range(5):
 p=O/f'k28_native_seed{seed}_review_results.json';a=read(p);reviews[str(p)]=sha(p);assert a['complete'] and a['seed']==seed and a['models']==1024
 for f,h in a['files'].items():assert f not in files and sha(P/f)==h;files[f]=h
 for n in names:totals[n].update(a['totals'][n])
 pairs+=a['paired_records'];baselines+=a['baseline_records'];reuses+=a['reuses'];cpu+=a['research_cpu'];maximum=max(maximum,a['max_shard_cpu'])
 ids={r['model_sha256'] for r in a['individual']};assert len(ids)==1024 and not ids&models;models|=ids
 individual += [{k:v for k,v in r.items() if k!='model_sha256'} for r in a['individual']]
 group=d['groups'][seed];assert group['seed']==seed
 for n in names:
  metric_equal(a['metrics'][n],group['recipes'][n])
  assert group['original_mean_counter_parity'][n]=={k:float(F(v)) for k,v in a['original_mean_counter_parity'][n].items()}
 for noise in range(20,24):
  row=d['noise_groups'][seed*4+noise-20];assert (row['seed'],row['noise'])==(seed,noise)
  for n in names:metric_equal(a['noise_metrics'][str(noise)][n],row['recipes'][n])
 for f in a['files']:
  if f.endswith('.json'):rss=max(rss,read(P/f)['rss_kib'])
 clean=[r for r in controls['rows'] if (r['seed'],r['noise'],r['copy'])==(seed,None,None)];assert len(clean)==1;c=clean[0];truth=rung['seeds'][str(seed)]['clean']['truth'];assert truth==c['truth'] and c['model_sha256']==recon['records'][seed]['base_sha256']
 assert d['clean'][seed]['seed']==seed
 for n in names:
  b,t=c['recipes'][n]['baseline'],c['recipes'][n]['traced'];old=margin['seeds'][str(seed)]['clean'][n]
  for key in ('hex','value','legacy','stats','info','recipe'):assert b[key]==t[key]
  assert b['hex']==float(b['value']).hex() and b['value']-truth==old['err'] and b['legacy']==old['ops'] and b['stats']['cells']==old['cells']
  count,num,integer=ledger(t['ledger']);ctotals[n].update(count)
  row=dict(value=b['value'],truth=truth,numeric=num,legacy=b['legacy'],seed=seed,noise=None,copy=None)
  result=metrics([row]);metric_equal(result,d['clean'][seed]['recipes'][n]);clean_results.append(dict(seed=seed,recipe=n,numeric=num,integer=integer,metrics=result))
assert files==d['files'] and len(files)==160 and len(models)==5120 and pairs==d['paired_records']==10240 and baselines==d['baseline_records']==10220
assert individual==d['individual'] and reuses==d['reused_controls']==[[s,n,j] for s in range(5) for n,j in ((20,0),(23,255))]
assert {n:dict(c) for n,c in totals.items()}==d['ledger_totals'] and {n:dict(c) for n,c in ctotals.items()}==d['clean_ledger_totals']
assert abs(cpu-d['research_cpu'])<1e-8 and cpu<=3000 and maximum==d['max_process_cpu']<=110 and rss==d['max_rss_kib']
losses={n:{conv:sum(r['ratios'][conv]>1 for r in individual if r['recipe']==n) for conv in ('named_numeric','historical')} for n in names}
out={'complete':True,'models':5120,'clean_models':5,'paired_records':pairs,'baseline_records':baselines,'files':160,'reuses':reuses,'totals':{n:dict(c) for n,c in totals.items()},'individual_losses':losses,'clean':clean_results,'research_cpu':cpu,'max_process_cpu':maximum,'review_cpu':time.process_time()-start,'seed_review_pins':reviews,'summary_sha256':sha(P/'full_summary.json'),'scope':'Global reconciliation reuses five completed independent raw-site audits, verifies clean ledgers and all exact stored-float group/noise/individual metrics. No evaluator/truth rerun. Historical checkpoint byte identity unproved; source allowances and conditional bootstrap are not full native/runtime or universal performance.'}
outpath.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('clean','totals','seed_review_pins')}))
