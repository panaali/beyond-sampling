"""Frozen original64-copy source-cost shard with once-only pilot reuse."""
import time
START=time.process_time()
from pathlib import Path
import gzip,hashlib,importlib.util,json,resource,signal,sys,traceback
HERE=Path(__file__).resolve().parent
RESERVE=110
TOTAL=3000
GRACE=90

def main(h,z,noise):
 assert h in (4,5,6) and 0<=z<5 and noise in (80,81,82,83)
 folder=HERE/'panel';folder.mkdir(exist_ok=True);stem=f'H{h}_z{z}_n{noise}';outpath=folder/f'{stem}.json';rawpath=folder/f'{stem}.jsonl.gz'
 assert not outpath.exists() and not rawpath.exists(),'Preserve every complete/partial attempt'
 gate=json.loads((HERE/'panel_gate.json').read_text())
 for path,pin in gate['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==pin,path
 prior=json.loads((HERE/'pilots.json').read_text());provenance=json.loads((HERE/'provenance.json').read_text())
 assert prior['complete'] and len(prior['rows'])==3 and provenance['complete']
 used=prior['cpu_seconds']+provenance['cpu_seconds']
 for path in sorted(folder.glob('H*_z*_n*.json')):
  prev=json.loads(path.read_text());used+=prev['cpu_seconds'] if prev['complete'] else prev['reservation_cpu']
 assert used+RESERVE<=TOTAL,('Research CPU reservation limit',used)
 manifest=json.loads((HERE/'manifest.json').read_text())
 output={'complete':False,'H':h,'zoo':z,'noise':noise,'rows':[],'reservation_cpu':RESERVE,'prior_charged_cpu':used,'total_research_cap':TOTAL,
  'gate_sha256':hashlib.sha256((HERE/'panel_gate.json').read_bytes()).hexdigest(),'source_calls_started':0,'source_calls_completed':0,'pilot_reuses':0,
  'sampler_numeric':4*h*h+12*h-1,'scope':'Original named-source convention; full per-site ledgers retained, no hardware/library/control/memory closure.'}
 def save():
  output.update(cpu_seconds=time.process_time()-START,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  tmp=outpath.with_suffix('.tmp');tmp.write_text(json.dumps(output,indent=2)+'\n');tmp.replace(outpath)
 save()
 def timeout(signum,frame):raise TimeoutError('105CPU soft cap; preserve partial shard')
 signal.signal(signal.SIGXCPU,timeout);resource.setrlimit(resource.RLIMIT_CPU,(105,110))
 try:
  for path,pin in manifest['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==pin,path
  native=Path(manifest['native_root']);sys.path.insert(0,str(native))
  spec=importlib.util.spec_from_file_location('t184_native_trace',native/'trace.py');T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
  assert Path(T.B.__file__).resolve()==native/'base_snapshot.py'
  import numpy as np
  import torch
  cp=next(c for c in manifest['checkpoints'] if (c['H'],c['zoo'])==(h,z));sd=torch.load(cp['path'],map_location='cpu',weights_only=True)
  base=(sd['rnn.weight_ih_l0'].numpy().astype(float)[:,0],sd['rnn.weight_hh_l0'].numpy().astype(float),sd['linear.weight'].numpy().astype(float))
  assert [w.shape for w in base]==[(h,),(h,h),(3,h)] and all(np.isfinite(w).all() for w in base)
  old=json.loads((HERE/f'frozen/core/core_{stem}.json').read_text());assert old['complete'] and len(old['rows'])==64
  output.update(checkpoint_sha256=cp['sha256'],prior_shard_sha256=hashlib.sha256((HERE/f'frozen/core/core_{stem}.json').read_bytes()).hexdigest(),
   source_sha256=hashlib.sha256(T.SOURCE.read_bytes()).hexdigest(),trace_sha256=hashlib.sha256((native/'trace.py').read_bytes()).hexdigest(),numpy_version=np.__version__,torch_version=torch.__version__)
  rng=np.random.default_rng(noise)
  with gzip.open(rawpath,'wt',compresslevel=6) as log:
   for idx,saved in enumerate(old['rows']):
    if time.process_time()-START>=GRACE:
     output['stopped']='90CPU graceful limit before next case';save();raise TimeoutError(output['stopped'])
    row={'copy':idx,'stage':'attempt','truth':saved['truth'],'saved_estimate':saved['saved_estimate']};output['rows'].append(row);save()
    log.write(json.dumps({'event':'attempt','copy':idx})+'\n');log.flush()
    model=tuple(w+.05*rng.standard_normal(w.shape) for w in base);sha=hashlib.sha256(b''.join(w.tobytes() for w in model)).hexdigest();assert sha==saved['model_sha256']
    row['model_sha256']=sha
    if (z,noise,idx)==(0,80,0):
     reuse=next(r for r in prior['rows'] if r['H']==h);assert reuse['model_sha256']==sha and reuse['parity']
     baseline,traced=reuse['baseline'],reuse['traced'];output['pilot_reuses']+=1;row['reused_pilot']=True
    else:
     output['source_calls_started']+=1;save();baseline=T.evaluate(model,False);output['source_calls_completed']+=1
     row['stage']='baseline_done';save();log.write(json.dumps({'event':'baseline','copy':idx,'baseline':baseline})+'\n');log.flush()
     output['source_calls_started']+=1;save();traced=T.evaluate(model,True);output['source_calls_completed']+=1;row['reused_pilot']=False
    log.write(json.dumps({'event':'evaluations','copy':idx,'baseline':baseline,'traced':traced})+'\n');log.flush()
    for key in ('hex','legacy','stats','info'):assert baseline[key]==traced[key],(idx,key)
    assert baseline['legacy']==saved['saved_ops'] and baseline['stats']['cells']==saved['saved_cells']
    assert baseline['hex']==float(saved['saved_estimate']).hex(),(idx,'saved source bits')
    row.update(stage='complete',parity=True,estimate=baseline['value'],hex=baseline['hex'],legacy=baseline['legacy'],cells=baseline['stats']['cells'],
     named_numeric=traced['ledger']['named_numeric_total'],integer_control=traced['ledger']['integer_control_total'],totals=traced['ledger']['totals'])
    save();log.write(json.dumps({'event':'complete','copy':idx,'model_sha256':sha})+'\n');log.flush()
  assert len(output['rows'])==64 and all(r['stage']=='complete' for r in output['rows'])
  assert output['source_calls_started']==output['source_calls_completed']==2*(64-output['pilot_reuses'])
  output.update(complete=True,archive_sha256=hashlib.sha256(rawpath.read_bytes()).hexdigest());save()
 except BaseException:
  output['failure']=traceback.format_exc();save();raise
 print(json.dumps({'shard':stem,'complete':True,'rows':64,'pilot_reuses':output['pilot_reuses'],'source_calls':output['source_calls_completed'],'cpu_seconds':output['cpu_seconds']}),flush=True)
if __name__=='__main__':main(*map(int,sys.argv[1:]))
