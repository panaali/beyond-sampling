"""Saved-only complete2500-block audit and fixed25-family reference intervals."""
import json,math,resource,time
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from scipy.stats import beta
import reference as R
HERE=Path(__file__).resolve().parent

def main():
 resource.setrlimit(resource.RLIMIT_CPU,(19,20));dst=HERE/'summary.json';assert not dst.exists();start=time.process_time()
 out=dict(complete=False,rows=[],block_pins={},attempt_pins={},source_sha256=R.sha(__file__))
 def save():dst.write_text(json.dumps(out,indent=2)+'\n')
 save()
 try:
  scope=R.check_scope('summary_scope.json');R.check_scope('continuation_scope.json');R.check_scope('source_scope.json');R.check_scope('pilot_scope.json')
  protocol=json.loads((HERE/'protocol.json').read_text());pilot=json.loads((HERE/'pilot.json').read_text());controls=json.loads((HERE/'controls.json').read_text())
  assert protocol['confidence_family']==25 and protocol['alpha']==.05 and controls['complete'] and pilot['complete']
  assert protocol['draws_per_model']==100000000 and protocol['blocks_per_model']==100 and protocol['block_size']==1000000
  assert len(protocol['models'])==25 and len({r['rng_seed'] for r in protocol['models']})==25
  assert controls['random_draws']==0 and controls['completed_extra_forward_evaluations']==318
  protocol_hash=R.sha(HERE/'protocol.json');forward_hash=R.sha(HERE/'reference.py');runner_hash=R.sha(HERE/'continue_reference.py');scope_hash=R.sha(HERE/'continuation_scope.json')
  prior_cpu=pilot['cpu_seconds']+controls['cpu_seconds'];total_process=F(0);forward_cpu=F(0);block_count=0
  delta=.05/(2*25)
  for index,ident in enumerate(protocol['models']):
   assert ident['index']==index and (ident['seed'],ident['copy'])==divmod(index,5)
   folder=HERE/'blocks'/f'{index:02d}';attempt_path=folder/'attempt.json';a=json.loads(attempt_path.read_text())
   assert a['complete'] and a['index']==index and a['completed_new_blocks']==(99 if index==0 else 100)
   assert a['pilot_reused']==(index==0) and a['new_draws']==(99 if index==0 else 100)*1000000
   assert a['source_sha256']==runner_hash and a['forward_source_sha256']==forward_hash and a['protocol_sha256']==protocol_hash and a['scope_sha256']==scope_hash
   assert a['model_sha256']==ident['model_sha256'] and a['numpy_version']==pilot['numpy_version']
   assert a['prior_attempt_hashes']=={str(i):out['attempt_pins'][str(i)] for i in range(index)}
   assert abs(a['prior_research_cpu_seconds']-prior_cpu)<1e-9 and prior_cpu+110<=1200 and a['reserved_cpu']==110
   assert math.isfinite(a['process_cpu_seconds']) and 0<=a['cpu_seconds']<=a['process_cpu_seconds']<110
   prior_cpu+=a['process_cpu_seconds'];total_process+=F(a['process_cpu_seconds']);out['attempt_pins'][str(index)]=R.sha(attempt_path)
   assert len(list(folder.glob('block*.json')))==(99 if index==0 else 100)
   state=np.random.Generator(np.random.PCG64(ident['rng_seed'])).bit_generator.state;hits=0;model_forward=F(0);previous_path=None;previous_sha=None
   for block in range(100):
    path=HERE/'pilot.json' if index==block==0 else folder/f'block{block:03d}.json'
    r=json.loads(path.read_text());assert r['complete'] and type(r['hits']) is int and 0<=r['hits']<=1000000 and r['n']==1000000
    assert r['index']==index and r['block']==block and (r['seed'],r['copy'])==divmod(index,5)
    assert r['model_sha256']==ident['model_sha256'] and r['protocol_sha256']==protocol_hash and r['numpy_version']==pilot['numpy_version']
    assert r['rng_seed']==ident['rng_seed'] and r['rng_name']==ident['rng_name'] and r['state_before']==state
    assert r['source_sha256']==(forward_hash if index==block==0 else runner_hash)
    if not index==block==0:
     assert r['forward_source_sha256']==forward_hash and r['previous_path']==(str(previous_path) if previous_path else None) and r['previous_sha256']==previous_sha
    state=r['state_after'];assert state['bit_generator']=='PCG64' and state['state']['inc']==r['state_before']['state']['inc']
    assert 0<=state['state']['state']<2**128 and state['has_uint32']==0
    assert math.isfinite(r['forward_cpu_seconds']) and r['forward_cpu_seconds']>=0
    model_forward+=F(r['forward_cpu_seconds']);hits+=r['hits'];block_count+=1
    previous_path=path;previous_sha=R.sha(path);out['block_pins'][f'{index}:{block}']=previous_sha
   assert a['last_block_sha256']==previous_sha
   n=100000000;lo=0. if hits==0 else float(beta.ppf(delta,hits,n-hits+1));hi=1. if hits==n else float(beta.ppf(1-delta,hits+1,n-hits))
   assert 0<=lo<=hits/n<=hi<=1
   out['rows'].append(dict(index=index,seed=ident['seed'],copy=ident['copy'],model_sha256=ident['model_sha256'],rng_seed=ident['rng_seed'],n=n,hits=hits,estimate=hits/n,lower=lo,upper=hi,forward_cpu_seconds=float(model_forward),forward_cpu_fraction=str(model_forward)))
   forward_cpu+=model_forward;save()
  assert len(out['rows'])==25 and block_count==2500 and len(out['block_pins'])==2500
  assert sum(r['n'] for r in out['rows'])==2500000000
  out.update(complete=True,draws=2500000000,blocks=2500,pilot_reuses=1,extra_deterministic_forwards=318,family_size=25,nominal_joint_coverage=.95,alpha=.05,interval_method='Clopper-Pearson Bonferroni25',numerical_beta_not_outward_rounded=True,forward_cpu_seconds=float(forward_cpu),forward_cpu_fraction=str(forward_cpu),continuation_process_cpu_seconds=float(total_process),continuation_process_cpu_fraction=str(total_process),recorded_research_cpu_seconds=prior_cpu,max_process_cpu_seconds=max(json.loads((HERE/'blocks'/f'{i:02d}'/'attempt.json').read_text())['process_cpu_seconds'] for i in range(25)),protocol_sha256=protocol_hash,pilot_sha256=R.sha(HERE/'pilot.json'),scope='Fixed fresh25-family references only; original stored-float forward/argsort semantics, no old count combination. Pilot/control imports excluded from recorded research timer; continuation process imports included. Source/runtime cost qualifications remain separate. No accuracy-based selection or estimator evaluation.')
 except BaseException as e:out['failure']=repr(e);raise
 finally:out['cpu_seconds']=time.process_time()-start;save()
 print(json.dumps({k:out[k] for k in ('complete','draws','blocks','forward_cpu_seconds','recorded_research_cpu_seconds')}))

if __name__=='__main__':main()
