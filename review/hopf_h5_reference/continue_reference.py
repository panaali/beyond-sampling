"""Fixed remaining 2,499 blocks; one model and one fair primary lock per process."""
import argparse,json,resource,signal,time
from pathlib import Path
import numpy as np
import reference as R
HERE=Path(__file__).resolve().parent

def expired(*args):raise TimeoutError('105CPU soft process cap')

def run(index):
 assert type(index) is int and 0<=index<25
 resource.setrlimit(resource.RLIMIT_CPU,(105,110));signal.signal(signal.SIGXCPU,expired)
 folder=HERE/'blocks'/f'{index:02d}';folder.mkdir(parents=True,exist_ok=True)
 dst=folder/'attempt.json';assert not dst.exists()
 start=time.process_time();out=dict(complete=False,index=index,source_sha256=R.sha(__file__),completed_new_blocks=0)
 def save():dst.write_text(json.dumps(out,indent=2)+'\n')
 save()
 try:
  scope=R.check_scope('continuation_scope.json');R.check_scope('source_scope.json');R.check_scope('pilot_scope.json')
  pilot=json.loads((HERE/'pilot.json').read_text());assert pilot['complete'] and pilot['projection_gate_pass'] and pilot['projected_full_forward_cpu_seconds']<=900
  assert type(pilot['hits']) is int and pilot['n']==1000000 and pilot['block']==pilot['index']==0
  assert np.__version__==pilot['numpy_version']
  controls=json.loads((HERE/'controls.json').read_text());assert controls['complete']
  prior_cpu=pilot['cpu_seconds']+controls['cpu_seconds'];prior_attempts={}
  for earlier in range(index):
   p=HERE/'blocks'/f'{earlier:02d}'/'attempt.json';a=json.loads(p.read_text())
   assert a['complete'] and a['index']==earlier and a['source_sha256']==R.sha(__file__)
   assert a['completed_new_blocks']==(99 if earlier==0 else 100)
   prior_cpu+=a['process_cpu_seconds'];prior_attempts[str(earlier)]=R.sha(p)
  out.update(prior_research_cpu_seconds=prior_cpu,prior_attempt_hashes=prior_attempts,reserved_cpu=110,research_envelope=1200,pilot_control_timer_scope='Recorded post-import CPU; their import/setup time excluded. Continuation process_cpu_seconds includes own imports.')
  save();assert prior_cpu+110<=1200,'research envelope exhausted; no new model started'
  protocol,old,new=R.load();model,h=R.model_for(protocol,index);ident=protocol['models'][index]
  assert protocol['blocks_per_model']==100 and protocol['block_size']==1000000 and protocol['draws_per_model']==100000000
  protocol_hash=R.sha(HERE/'protocol.json');forward_hash=R.sha(HERE/'reference.py')
  out.update(model_sha256=h,protocol_sha256=protocol_hash,forward_source_sha256=forward_hash,scope_sha256=R.sha(HERE/'continuation_scope.json'),numpy_version=np.__version__,seed=ident['seed'],copy=ident['copy'])
  save()
  first=1 if index==0 else 0
  for block in range(first,100):
   path=folder/f'block{block:03d}.json';assert not path.exists()
   rng=np.random.Generator(np.random.PCG64(ident['rng_seed']));previous_path=None;previous_sha=None
   if block:
    previous_path=HERE/'pilot.json' if index==0 and block==1 else folder/f'block{block-1:03d}.json'
    previous=json.loads(previous_path.read_text())
    assert previous['complete'] and previous['index']==index and previous['block']==block-1
    assert previous['model_sha256']==h and previous['protocol_sha256']==protocol_hash and previous['numpy_version']==np.__version__
    assert previous['rng_seed']==ident['rng_seed'] and previous['rng_name']==ident['rng_name']
    assert type(previous['hits']) is int and 0<=previous['hits']<=1000000 and previous['n']==1000000
    if previous_path.name=='pilot.json':assert previous['source_sha256']==forward_hash
    else:assert previous['source_sha256']==R.sha(__file__) and previous['forward_source_sha256']==forward_hash
    previous_sha=R.sha(previous_path);rng.bit_generator.state=previous['state_after']
   row=dict(complete=False,index=index,seed=ident['seed'],copy=ident['copy'],block=block,n=1000000,model_sha256=h,protocol_sha256=protocol_hash,source_sha256=R.sha(__file__),forward_source_sha256=forward_hash,numpy_version=np.__version__,rng_seed=ident['rng_seed'],rng_name=ident['rng_name'],previous_path=str(previous_path) if previous_path else None,previous_sha256=previous_sha,state_before=rng.bit_generator.state,phase='forward_started')
   def save_row():path.write_text(json.dumps(row,indent=2)+'\n')
   save_row();t=time.process_time()
   try:
    hits=new(model,1000000,rng,chunk=1000000)
    assert type(hits) is int and 0<=hits<=1000000
    row.update(complete=True,phase='complete',hits=hits,state_after=rng.bit_generator.state)
   except BaseException as e:row.update(phase='failed',failure=repr(e));raise
   finally:row['forward_cpu_seconds']=time.process_time()-t;save_row()
   out['completed_new_blocks']+=1;out['last_block_sha256']=R.sha(path);save()
  assert out['completed_new_blocks']==100-first
  out.update(complete=True,pilot_reused=(index==0),new_draws=(100-first)*1000000)
 except BaseException as e:out.update(failure=repr(e),failed_process_reservation=110);raise
 finally:
  out.update(cpu_seconds=time.process_time()-start,process_cpu_seconds=time.process_time(),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save()
 print(json.dumps({k:out[k] for k in ('complete','index','completed_new_blocks','process_cpu_seconds')}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);run(p.parse_args().index)
