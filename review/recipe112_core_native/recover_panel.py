"""Published T184 recovery: distinct immutable records, no repeated evaluations."""
import time
START=time.process_time()
from pathlib import Path
import gzip,hashlib,importlib.util,json,resource,signal,sys,traceback
HERE=Path(__file__).resolve().parent
RESERVE=110;TOTAL=3000;GRACE=90

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(h,z,noise):
 stem=f'H{h}_z{z}_n{noise}';gate=json.loads((HERE/'recovery_gate.json').read_text());assert stem in gate['remaining_shards']
 for p,s in gate['pins'].items():assert sha(p)==s,p
 folder=HERE/'panel_v2';folder.mkdir(exist_ok=True);outpath=folder/f'{stem}.json';rawpath=folder/f'{stem}.jsonl.gz'
 assert not outpath.exists() and not rawpath.exists(),'Never overwrite complete or partial recovery'
 prior=json.loads((HERE/'pilots.json').read_text());prov=json.loads((HERE/'provenance.json').read_text())
 used=prior['cpu_seconds']+prov['cpu_seconds'];old_complete=[];old_failures=[]
 for p in sorted((HERE/'panel').glob('H*_z*_n*.json')):
  d=json.loads(p.read_text())
  if d['complete']:old_complete.append(p.stem);used+=d['cpu_seconds']
  else:
   assert p.stem==gate['accepted_failure'] and sha(p)==gate['pins'][str(p.resolve())]
   old_failures.append(p.stem);used+=d['reservation_cpu']
 assert old_complete==gate['completed_shards'] and old_failures==[gate['accepted_failure']]
 for p in sorted(folder.glob('H*_z*_n*.json')):
  d=json.loads(p.read_text());used+=d['cpu_seconds'] if d['complete'] else d['reservation_cpu']
  assert d['complete'],('A new incomplete recovery blocks all later work',str(p))
 assert used+RESERVE<=TOTAL,('Research reservation limit',used)
 manifest=json.loads((HERE/'manifest.json').read_text());recovered=json.loads((HERE/'recovery_evaluations.json').read_text())
 output={'complete':False,'H':h,'zoo':z,'noise':noise,'rows':[],'reservation_cpu':RESERVE,'prior_charged_cpu':used,'total_research_cap':TOTAL,
  'gate_sha256':sha(HERE/'recovery_gate.json'),'source_calls_started':0,'source_calls_completed':0,'pilot_reuses':0,'failed_attempt_reuses':0,
  'sampler_numeric':4*h*h+12*h-1,'scope':'Published native/scalar counter-provenance recovery. Same native output/legacy and baseline/traced all-field parity; scalar topology diagnostic separately.'}
 def save():
  output.update(cpu_seconds=time.process_time()-START,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  tmp=outpath.with_suffix('.tmp');tmp.write_text(json.dumps(output,indent=2)+'\n');tmp.replace(outpath)
 save()
 def timeout(signum,frame):raise TimeoutError('105CPU soft cap; preserve partial recovery')
 signal.signal(signal.SIGXCPU,timeout);resource.setrlimit(resource.RLIMIT_CPU,(105,110))
 try:
  for p,s in manifest['pins'].items():assert sha(p)==s,p
  native=Path(manifest['native_root']);sys.path.insert(0,str(native));spec=importlib.util.spec_from_file_location('t184_native_trace',native/'trace.py');T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
  assert Path(T.B.__file__).resolve()==native/'base_snapshot.py'
  import numpy as np
  import torch
  cp=next(c for c in manifest['checkpoints'] if (c['H'],c['zoo'])==(h,z));sd=torch.load(cp['path'],map_location='cpu',weights_only=True)
  base=(sd['rnn.weight_ih_l0'].numpy().astype(float)[:,0],sd['rnn.weight_hh_l0'].numpy().astype(float),sd['linear.weight'].numpy().astype(float))
  assert [w.shape for w in base]==[(h,),(h,h),(3,h)] and all(np.isfinite(w).all() for w in base)
  oldpath=HERE/f'frozen/core/core_{stem}.json';old=json.loads(oldpath.read_text());assert old['complete'] and len(old['rows'])==64
  output.update(checkpoint_sha256=cp['sha256'],prior_shard_sha256=sha(oldpath),source_sha256=sha(T.SOURCE),trace_sha256=sha(native/'trace.py'),numpy_version=np.__version__,torch_version=torch.__version__)
  rng=np.random.default_rng(noise)
  with gzip.open(rawpath,'wt',compresslevel=6) as log:
   for idx,saved in enumerate(old['rows']):
    if time.process_time()-START>=GRACE:raise TimeoutError('90CPU graceful limit before next recovery case')
    row={'copy':idx,'stage':'attempt','truth':saved['truth'],'saved_estimate':saved['saved_estimate']};output['rows'].append(row);save()
    log.write(json.dumps({'event':'attempt','copy':idx})+'\n');log.flush()
    model=tuple(w+.05*rng.standard_normal(w.shape) for w in base);digest=hashlib.sha256(b''.join(w.tobytes() for w in model)).hexdigest();assert digest==saved['model_sha256'];row['model_sha256']=digest
    if stem==gate['accepted_failure'] and idx in (0,1):
     reuse=recovered['rows'][idx];assert reuse['copy']==idx and reuse['model_sha256']==digest
     baseline,traced=reuse['baseline'],reuse['traced'];output['failed_attempt_reuses']+=1;row['reused_from']='failed_v1_saved_evaluations'
    elif (h,z,noise,idx)==(6,0,80,0):
     reuse=next(r for r in prior['rows'] if r['H']==h);assert reuse['model_sha256']==digest and reuse['parity']
     baseline,traced=reuse['baseline'],reuse['traced'];output['pilot_reuses']+=1;row['reused_from']='pilot'
    else:
     assert not (z==0 and noise==80 and idx==0),'Only remaining H6 pilot may be reused here'
     output['source_calls_started']+=1;save();baseline=T.evaluate(model,False);output['source_calls_completed']+=1;row['stage']='baseline_done';save()
     log.write(json.dumps({'event':'baseline','copy':idx,'baseline':baseline})+'\n');log.flush()
     output['source_calls_started']+=1;save();traced=T.evaluate(model,True);output['source_calls_completed']+=1;row['reused_from']=None
    log.write(json.dumps({'event':'evaluations','copy':idx,'baseline':baseline,'traced':traced,'reused_from':row['reused_from']})+'\n');log.flush()
    for key in ('hex','legacy','stats','info'):assert baseline[key]==traced[key],(idx,key)
    assert baseline['legacy']==saved['saved_ops'] and baseline['hex']==float(saved['saved_estimate']).hex()
    row.update(stage='complete',parity=True,estimate=baseline['value'],hex=baseline['hex'],legacy=baseline['legacy'],cells=baseline['stats']['cells'],native_stats=baseline['stats'],
     saved_scalar_cells=saved['saved_cells'],saved_scalar_stats=saved['stats'],scalar_native_cells_equal=baseline['stats']['cells']==saved['saved_cells'],
     named_numeric=traced['ledger']['named_numeric_total'],integer_control=traced['ledger']['integer_control_total'],totals=traced['ledger']['totals'])
    save();log.write(json.dumps({'event':'complete','copy':idx,'model_sha256':digest})+'\n');log.flush()
  assert len(output['rows'])==64 and all(r['stage']=='complete' for r in output['rows'])
  assert output['pilot_reuses']==sum(r['reused_from']=='pilot' for r in output['rows'])
  assert output['failed_attempt_reuses']==sum(r['reused_from']=='failed_v1_saved_evaluations' for r in output['rows'])
  assert output['source_calls_started']==output['source_calls_completed']==2*(64-output['pilot_reuses']-output['failed_attempt_reuses'])
  output.update(complete=True,archive_sha256=sha(rawpath));save()
 except BaseException:
  output['failure']=traceback.format_exc();save();raise
 print(json.dumps({'shard':stem,'complete':True,'rows':64,'pilot_reuses':output['pilot_reuses'],'failed_attempt_reuses':output['failed_attempt_reuses'],'source_calls':output['source_calls_completed'],'cpu_seconds':output['cpu_seconds']}),flush=True)
if __name__=='__main__':main(*map(int,sys.argv[1:]))
