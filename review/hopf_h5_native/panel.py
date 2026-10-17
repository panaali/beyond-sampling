"""Remaining fixed original H5 h64 trace after pilot gate."""
import argparse,gzip,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,choices=range(1,25),required=True);args=ap.parse_args();seed,copy=divmod(args.index,5)
 resource.setrlimit(resource.RLIMIT_CPU,(105,110))
 folder=HERE/'panel';folder.mkdir(exist_ok=True);dst=folder/f'{args.index:02d}.json';raw=folder/f'{args.index:02d}_ledger.json.gz';assert not dst.exists() and not raw.exists()
 out={'complete':False,'seed':seed,'copy':copy,'index':args.index,'nu':64,'nd':128,'source_sha256':sha(__file__),'scope':'One of the fixed remaining24 original saved H5 cases. Named arithmetic and conditional source sort/search allowance only; no full native/runtime, panel, root-mask repair or reference precision transfer.'}
 def save():dst.write_text(json.dumps(out,indent=2)+'\n')
 save();start=time.process_time()
 try:
  scope=json.loads((HERE/'panel_scope.json').read_text())
  for p,h in scope['pins'].items():assert sha(p)==h,p
  prior=json.loads((HERE/'pilot.json').read_text());assert prior['complete'] and prior['seed']==prior['copy']==0
  protocol=json.loads((HERE/'protocol.json').read_text())
  for p,h in {**protocol['source_pins'],**protocol['frozen_pins']}.items():assert sha(p)==h,p
  controls=json.loads((HERE/'controls.json').read_text());assert controls['complete'] and len(controls['rows'])==8
  models=json.loads((HERE/'models.json').read_text());assert models['complete']
  saved=next(r for r in models['models'] if (r['seed'],r['copy'])==(seed,copy));model=tuple(np.array(x,dtype=float) for x in saved['model'])
  assert [x.shape for x in model]==[(5,),(5,5),(4,5)]
  mh=hashlib.sha256(b''.join(x.astype('<f8').tobytes() for x in model)).hexdigest();assert mh==saved['model_sha256']
  root=Path(protocol['tracer_root']);sys.path.insert(0,str(root));import tracer,closed_tracer
  assert Path(tracer.__file__).resolve()==root/'tracer.py' and Path(closed_tracer.__file__).resolve()==root/'closed_tracer.py'
  out.update(model_sha256=mh,saved_value=saved['saved_h64'],saved_hex=saved['saved_h64_hex'],saved_legacy=saved['saved_legacy'],phase='baseline_started');save()
  base,_,_=tracer.load(False,False);bo=tracer.Ops();baseline=base.est_hopf(model,64,bo,nd=128)
  out.update(baseline=baseline,baseline_hex=float(baseline).hex(),baseline_legacy=bo.n,baseline_cpu_seconds=time.process_time()-start);save()
  assert float(baseline).hex()==saved['saved_h64_hex'] and bo.n==saved['saved_legacy'],'historical original parity failed'
  out['phase']='trace_started';save()
  mod,ledger,startup=closed_tracer.load(False);ops=tracer.Ops();value=mod.est_hopf(model,64,ops,nd=128)
  report=ledger.report();out.update(value=value,value_hex=float(value).hex(),legacy=ops.n)
  # Preserve the full raw trace even when the final output assertion fails.
  with gzip.open(raw,'wt') as f:json.dump({'report':report,'startup':startup},f)
  out.update(ledger_sha256=sha(raw),explicit_with_constant_setup=report['inclusive_explicit_total']+startup['inclusive_explicit_total'],conditional_source_with_constant_setup=report['explicit_plus_conditional_source_allowance']+startup['inclusive_explicit_total'],startup_explicit=startup['inclusive_explicit_total'],opaque_calls=report['opaque_calls'],source_closure_count=len(report['source_closures']))
  save();assert out['value_hex']==out['baseline_hex'] and ops.n==bo.n
  assert sum(r['count'] for r in report['opaque_calls'])==len(report['source_closures'])
  out.update(complete=True,phase='complete')
 except BaseException as e:out['failure']=repr(e);raise
 finally:
  out.update(cpu_seconds=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save()
 print(json.dumps({k:out[k] for k in ('complete','index','value','legacy','explicit_with_constant_setup','conditional_source_with_constant_setup','cpu_seconds')}))
if __name__=='__main__':main()
