"""Frozen tuple semantics and unchanged-source RECIPE85B controls."""
import json,time,traceback
from pathlib import Path
import numpy as np
import native as N
HERE=Path(__file__).resolve().parent

def sort_case(name,items,expected_numeric,reverse=False):
 baseline=sorted(items,reverse=reverse);ledger=N.Ledger();traced=ledger.sort(name,sorted,items,reverse=reverse)
 assert all(a is b for a,b in zip(baseline,traced)) and len(baseline)==len(traced)
 r=ledger.report();assert r['named_numeric_total']==expected_numeric,(name,r,expected_numeric)
 return dict(name=name,reverse=reverse,order=[next(i for i,x in enumerate(items) if x is v) for v in traced],numeric=r['named_numeric_total'],ledger=r)

def main():
 start=time.process_time();rows=[]
 rows.append(sort_case('unequal_float_tuples',[(np.float64(2.),0),(np.float64(1.),1)],2))
 rows.append(sort_case('equal_distinct_float_tuples',[(np.float64(1.),1),(np.float64(1.),0)],1))
 shared=np.float64(1.);rows.append(sort_case('shared_identity',[(shared,1),(shared,0)],0))
 rows.append(sort_case('prefix',[(shared,0),(shared,)],0))
 rows.append(sort_case('scalar_keys',[np.float64(2.),np.float64(1.)],1))
 shared_nan=np.float64('nan');rows.append(sort_case('shared_nan_identity',[(shared_nan,1),(shared_nan,0)],0))
 rows.append(sort_case('distinct_nan',[(np.float64('nan'),1),(np.float64('nan'),0)],2))
 rows.append(sort_case('reverse',[(np.float64(2.),0),(np.float64(1.),1)],2,True))
 rows.append(sort_case('stable_equal',[(shared,0),(shared,0)],0))
 models=[]
 for name,wi,wh,wo in [('zero',[0.],[[0.]],[[0.],[0.],[0.]]),('zero_readout',[1.],[[0.]],[[0.],[0.],[0.]]),('identical_readout',[1.],[[0.]],[[1.],[1.],[1.]]),('dominant_first',[1.],[[0.]],[[1.],[0.],[-1.]]),('mixed',[1.,-1.],[[0.,1.],[1.,0.]],[[1.,0.],[0.,1.],[-1.,0.]])]:
  m=tuple(np.asarray(x) for x in (wi,wh,wo));plain=N.evaluate(m,False);traced=N.evaluate(m,True)
  for key in ('hex','legacy','stats','info'):assert plain[key]==traced[key],(name,key)
  assert any(site.startswith('_resolve_inv:') for site in traced['ledger']['sites'])
  models.append(dict(name=name,baseline=plain,traced=traced))
 out=dict(sort_cases=rows,models=models,cpu=time.process_time()-start,scope='Named-source accounting controls only; original source tie failures retained.')
 (HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(sort_cases=len(rows),model_cases=len(models),cpu=out['cpu'],native=[r['traced']['ledger']['named_numeric_total'] for r in models],sort_calls=[r['traced']['ledger']['sort_dispatch_calls'] for r in models])))
if __name__=='__main__':
 try:main()
 except Exception as exc:
  (HERE/'controls_failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2)+'\n');raise
