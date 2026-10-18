"""Independent complete 25-model/2,500-block saved reference audit. No redraws."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math,time,resource
import numpy as np
from scipy.special import betainc,betaincinv,betaincc
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
P=Path('/workspace/work/recover-small_h2_910/review/hopf_h5_reference');O=Path(__file__).resolve().parent;target=O/'h5_reference_complete_review_results.json';assert not target.exists()
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=read(P/'summary.json');protocol=read(P/'protocol.json');pilot=read(P/'pilot.json');controls=read(P/'controls.json');assert s['complete']
for filename in ('summary_scope.json','continuation_scope.json','source_scope.json','pilot_scope.json'):
 for p,h in read(P/filename)['pins'].items():assert sha(p)==h,p
for r in protocol['files'].values():assert sha(r['path'])==r['sha256']
assert len(protocol['models'])==25 and len({r['rng_seed'] for r in protocol['models']})==25
assert s['family_size']==25 and s['draws']==2500000000 and s['blocks']==2500 and s['pilot_reuses']==1 and s['extra_deterministic_forwards']==318
assert s['source_sha256']==sha(P/'summarize.py') and s['protocol_sha256']==sha(P/'protocol.json') and s['pilot_sha256']==sha(P/'pilot.json')
assert s['numerical_beta_not_outward_rounded'] and s['nominal_joint_coverage']==.95
source=sha(P/'reference.py');runner=sha(P/'continue_reference.py');prothash=sha(P/'protocol.json');scopehash=sha(P/'continuation_scope.json')
blocks={};attempts={};rows=[];totalforward=F();totalprocess=F();prior=pilot['cpu_seconds']+controls['cpu_seconds'];max_tail_error=0.;max_endpoint_difference=0.;unique_models=set()
assert len(s['rows'])==25
for idx,ident in enumerate(protocol['models']):
 assert ident['index']==idx and (ident['seed'],ident['copy'])==divmod(idx,5) and ident['model_sha256'] not in unique_models;unique_models.add(ident['model_sha256'])
 folder=P/'blocks'/f'{idx:02d}';attempt=read(folder/'attempt.json');expected_new=99 if idx==0 else 100
 assert attempt['complete'] and attempt['index']==idx and attempt['completed_new_blocks']==expected_new and attempt['pilot_reused']==(idx==0) and attempt['new_draws']==expected_new*1000000
 assert attempt['source_sha256']==runner and attempt['forward_source_sha256']==source and attempt['protocol_sha256']==prothash and attempt['scope_sha256']==scopehash
 assert attempt['model_sha256']==ident['model_sha256'] and attempt['prior_attempt_hashes']==attempts
 assert abs(attempt['prior_research_cpu_seconds']-prior)<1e-9 and prior+110<=1200 and attempt['reserved_cpu']==110
 assert math.isfinite(attempt['process_cpu_seconds']) and 0<=attempt['cpu_seconds']<=attempt['process_cpu_seconds']<110
 prior+=attempt['process_cpu_seconds'];totalprocess+=F(attempt['process_cpu_seconds']);attempts[str(idx)]=sha(folder/'attempt.json')
 expectedfiles={f'block{i:03d}.json' for i in range(1 if idx==0 else 0,100)};assert {p.name for p in folder.glob('block*.json')}==expectedfiles
 state=np.random.PCG64(ident['rng_seed']).state;lastpath=None;lastsha=None;hits=0;forward=F()
 for block in range(100):
  firstpilot=idx==0 and block==0;path=P/'pilot.json' if firstpilot else folder/f'block{block:03d}.json';r=read(path)
  assert r['complete'] and r['phase']=='complete' and r['index']==idx and r['block']==block and (r['seed'],r['copy'])==divmod(idx,5)
  assert r['n']==1000000 and type(r['hits'])is int and 0<=r['hits']<=r['n']
  assert r['model_sha256']==ident['model_sha256'] and r['rng_seed']==ident['rng_seed'] and r['rng_name']==ident['rng_name'] and r['protocol_sha256']==prothash and r['numpy_version']==pilot['numpy_version']
  assert r['source_sha256']==(source if firstpilot else runner) and r['state_before']==state
  if not firstpilot:assert r['forward_source_sha256']==source and r['previous_path']==(str(lastpath) if lastpath else None) and r['previous_sha256']==lastsha
  end=r['state_after'];assert end['bit_generator']=='PCG64' and end['state']['inc']==state['state']['inc'] and 0<=end['state']['state']<2**128 and end['state']['state']!=state['state']['state'] and end['has_uint32']==0
  assert math.isfinite(r['forward_cpu_seconds']) and r['forward_cpu_seconds']>=0
  state=end;lastpath=path;lastsha=sha(path);blocks[f'{idx}:{block}']=lastsha;hits+=r['hits'];forward+=F(r['forward_cpu_seconds'])
 assert attempt['last_block_sha256']==lastsha
 actual=s['rows'][idx];assert actual['index']==idx and actual['hits']==hits and actual['n']==100000000 and actual['estimate']==hits/100000000 and actual['model_sha256']==ident['model_sha256']
 assert F(actual['forward_cpu_fraction'])==forward and actual['forward_cpu_seconds']==float(forward)
 N=100000000;delta=.05/50
 lo=0. if hits==0 else float(betaincinv(hits,N-hits+1,delta));hi=1. if hits==N else float(betaincinv(hits+1,N-hits,1-delta))
 assert 0<=actual['lower']<=hits/N<=actual['upper']<=1
 max_endpoint_difference=max(max_endpoint_difference,abs(lo-actual['lower']),abs(hi-actual['upper']))
 assert abs(lo-actual['lower'])<1e-12 and abs(hi-actual['upper'])<1e-12
 residual=max(abs(float(betainc(hits,N-hits+1,actual['lower']))-delta) if hits else 0.,abs(float(betaincc(hits+1,N-hits,actual['upper']))-delta) if hits<N else 0.)
 assert residual<1e-7;max_tail_error=max(max_tail_error,residual)
 rows.append(dict(index=idx,hits=hits,n=N,lower=actual['lower'],upper=actual['upper'],tail_residual=residual,model_sha256=ident['model_sha256']));totalforward+=forward
assert blocks==s['block_pins'] and attempts==s['attempt_pins'] and len(blocks)==2500
assert F(s['forward_cpu_fraction'])==totalforward and F(s['continuation_process_cpu_fraction'])==totalprocess and abs(s['recorded_research_cpu_seconds']-prior)<1e-9
out={'pass':True,'models':25,'blocks':2500,'draws':2500000000,'extra_deterministic_forwards':318,'pilot_reuses':1,'rows':rows,'summary_sha256':sha(P/'summary.json'),'protocol_sha256':prothash,'source_sha256':source,'runner_sha256':runner,'forward_cpu':float(totalforward),'recorded_research_cpu':prior,'max_beta_tail_residual':max_tail_error,'max_endpoint_difference':max_endpoint_difference,'review_cpu':time.process_time()-start,'review_draws':0,'scope':'Complete saved coverage, independent initial PCG states, hash/state chains, integer-hit sums, resource ledger and fixed Bonferroni25 CP formulas. End states not replayed; beta inversions ordinary numerical, not outward proof. No estimator/model/reference redraws; comparison awaits separate audit.'}
target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
