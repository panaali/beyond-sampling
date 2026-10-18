"""First saved noisy h64 row; unchanged original numerical source and observer."""
import argparse,gzip,hashlib,json,resource,time
from pathlib import Path
import numpy as np
from tracer import HERE,ROOT,Ops,load as baseline_load
from closed_tracer import load
from panel_models import digest,model_hash

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--copy',type=int,required=True);args=ap.parse_args()
 assert (args.seed,args.copy)==(0,1),'Only frozen first-noise pilot authorized by this entrypoint'
 dst=HERE/f'panel_s{args.seed}_c{args.copy}.json.gz';assert not dst.exists()
 required=['controls.json','sort_controls.json','argsort4_controls.json','closed_controls.json']
 for name in required:assert json.loads((HERE/name).read_text())['status']=='complete'
 assert digest(HERE/'sort_shadow.cpp')==json.loads((HERE/'sort_controls.json').read_text())['cpp_sha256']
 metadata=json.loads((HERE/'panel_models.json').read_text());assert metadata['noise_source_sha256']==digest(HERE/'noise_snapshot.py') and metadata['checkpoint_identity_sha256']==digest(HERE/'panel_checkpoint_identity.json')
 row=next(r for r in metadata['rows'] if (r['seed'],r['copy'])==(args.seed,args.copy));model=tuple(np.array(w,dtype=np.float64) for w in row['model']);assert model_hash(model)==row['model_sha256']
 names=required+['tracer.py','closed_tracer.py','sort_shadow.cpp','sort_shadow.so','panel_models.py','panel_models.json','panel_checkpoint_identity.json','noise_snapshot.py','panel_probe.py']
 pins={name:digest(HERE/name) for name in names};pins['frozen_ev_44.json']=digest(ROOT/'frozen/ev_44.json')
 saved=json.loads((ROOT/'frozen/ev_44.json').read_text())[str(args.seed)]['rows'][args.copy]
 out={'status':'running','claim':'304a1ddb','seed':args.seed,'copy':args.copy,'arm':'original','nu':64,'nd':128,'pins':pins,'model_sha256':row['model_sha256'],'saved_value':saved['h64'],'saved_legacy':saved['h64_ops'],'scope':'Original-source existing fixed model only. Conditional mixed-category source allowances, no full runtime or h128 truth claim.'}
 def save():dst.write_bytes(gzip.compress((json.dumps(out,indent=2)+'\n').encode(),mtime=0))
 save();start=time.process_time()
 try:
  baseline,_,_=baseline_load(False,False);bo=Ops();base_value=baseline.est_hopf(model,64,bo,nd=128)
  out.update(baseline_value=base_value,baseline_legacy=bo.n);save();assert base_value==saved['h64'] and bo.n==saved['h64_ops'],'original saved parity failure'
  mod,ledger,startup=load(False);ops=Ops();value=mod.est_hopf(model,64,ops,nd=128);assert value==base_value and ops.n==bo.n
  report=ledger.report();cold=startup['inclusive_explicit_total'];assert sum(r['count'] for r in report['opaque_calls'])==len(report['source_closures'])
  out.update(status='complete',value=value,legacy=ops.n,ledger=report,startup=startup,explicit_with_constant_setup=report['inclusive_explicit_total']+cold,conditional_source_with_constant_setup=report['explicit_plus_conditional_source_allowance']+cold,cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save()
  print(json.dumps({k:out[k] for k in ('status','seed','copy','value','legacy','explicit_with_constant_setup','conditional_source_with_constant_setup','cpu_seconds')}))
 except Exception as e:out.update(status='failed',failure=repr(e),cpu_seconds=time.process_time()-start);save();raise
if __name__=='__main__':main()
