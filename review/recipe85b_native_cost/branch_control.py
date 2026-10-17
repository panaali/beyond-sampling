"""Frozen rational integrated _resolve_inv sorting control."""
import json,time,traceback
from pathlib import Path
import numpy as np
import native as N
HERE=Path(__file__).resolve().parent

def main():
 start=time.process_time();model=(np.array([1.,2.,-1.]),np.array([[1.,-2.,0.],[-3.,1.,1.],[2.,0.,-1.]]),np.array([[1.,0.,-1.],[-1.,2.,0.],[0.,-1.,1.]]))
 plain=N.evaluate(model,False);traced=N.evaluate(model,True)
 for key in ('hex','legacy','stats','info'):assert plain[key]==traced[key],key
 out=dict(model=[m.tolist() for m in model],baseline=plain,traced=traced,sort_coverage=traced['ledger']['sort_dispatch_calls']>0,cpu=time.process_time()-start)
 (HERE/'branch_control.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(value=plain['value'],sort_calls=traced['ledger']['sort_dispatch_calls'],native=traced['ledger']['named_numeric_total'],cpu=out['cpu'])))
 assert out['sort_coverage'],'Preserve insufficient sorting coverage'
if __name__=='__main__':
 try:main()
 except Exception as exc:
  (HERE/'branch_failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2)+'\n');raise
