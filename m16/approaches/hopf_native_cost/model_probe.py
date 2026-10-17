"""One predeclared existing seed0 H4T4 h8 source-cost trace, two fixed arms."""
import argparse,hashlib,json,time,resource
from pathlib import Path
import numpy as np
from tracer import HERE,ROOT,Ops,load as baseline_load
from closed_tracer import load
from model_identity import weights

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['original','tile'],required=True);args=ap.parse_args();rescue=args.arm=='tile'
 dst=HERE/f'model_probe_{args.arm}.json';assert not dst.exists()
 required=['controls.json','sort_controls.json','argsort4_controls.json','closed_controls.json']
 for name in required:assert json.loads((HERE/name).read_text())['status']=='complete'
 assert digest(HERE/'sort_shadow.cpp')==json.loads((HERE/'sort_controls.json').read_text())['cpp_sha256']
 # The numerical source and all prior independently gated controls are pinned.
 pin_names=required+['tracer.py','closed_tracer.py','sort_shadow.cpp','sort_shadow.so','model_identity.py','model_identity.json','checkpoint_identity.json']
 pins={name:digest(HERE/name) for name in pin_names};pins['frozen_ev_44.json']=digest(ROOT/'frozen/ev_44.json')
 model=weights();identity=json.loads((HERE/'model_identity.json').read_text());model_hash=hashlib.sha256(b''.join(w.astype('<f8').tobytes() for w in model)).hexdigest();assert model_hash==identity['float64_c_order_sha256']
 saved=json.loads((ROOT/'frozen/ev_44.json').read_text())['0']['rows'][0]
 out={'status':'running','claim':'fec66d8b','arm':args.arm,'H':4,'T':4,'seed':0,'nu':8,'nd':16,'model_sha256':model_hash,'pins':pins,'saved_original_value':saved['h8'],'saved_original_legacy':saved['h8_ops'],'scope':'Single existing h8 model/recipe discriminator. No h64/panel transfer, no new MC, no full runtime cost; source allowances retain binary/control/memory qualifications.'}
 save=lambda:dst.write_text(json.dumps(out,indent=2)+'\n');save();start=time.process_time()
 try:
  baseline,_,_=baseline_load(False,rescue);bo=Ops();base_value=baseline.est_hopf(model,8,bo,nd=16)
  out.update(baseline_value=base_value,baseline_legacy=bo.n);save()
  if not rescue:assert base_value==saved['h8'] and bo.n==saved['h8_ops'],'original saved parity failure'
  mod,ledger,startup=load(rescue);ops=Ops();value=mod.est_hopf(model,8,ops,nd=16);assert value==base_value and ops.n==bo.n
  report=ledger.report();cold=startup['inclusive_explicit_total']
  out.update(status='complete',value=value,legacy=ops.n,ledger=report,startup=startup,explicit_with_constant_setup=report['inclusive_explicit_total']+cold,conditional_source_with_constant_setup=report['explicit_plus_conditional_source_allowance']+cold,cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  assert sum(r['count'] for r in report['opaque_calls'])==len(report['source_closures']);save()
  print(json.dumps({k:out[k] for k in ('status','arm','value','legacy','explicit_with_constant_setup','conditional_source_with_constant_setup','cpu_seconds')}))
 except Exception as e:out.update(status='failed',failure=repr(e),cpu_seconds=time.process_time()-start);save();raise
if __name__=='__main__':main()
