import time
START=time.process_time()
from pathlib import Path
from fractions import Fraction as F
import json,resource
import primitive as A
P=Path(__file__).resolve().parent

def main():
 resource.setrlimit(resource.RLIMIT_CPU,(20,22));bound=A.Bound();source=json.loads((P.parent/'controls.json').read_text());rows=[]
 for r in source['rows']:
  if r['mode']!='bounded':continue
  out=bound.one(r);out.update(name=r['name'],power=r['power'],previous_mp_bound=r['quadrature']['bound_decimal']);assert F(out['upper_bound'])>=F(out['previous_mp_bound']);rows.append(out)
 constant=bound.one(dict(a='1',b='1',det='1'));assert F(constant['upper_bound'])==0
 try:bound.one(dict(a='0',b='1',det='1'))
 except ValueError as exc:invalid=dict(rejected=True,reason=repr(exc))
 else:raise AssertionError('invalid radial range accepted')
 out=dict(complete=True,rows=rows,constant=constant,invalid=invalid,cold_counts=bound.cold_counts,all_counts=dict(bound.D.counts),cpu=time.process_time()-START)
 assert not (P/'controls.json').exists();(P/'controls.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(complete=True,cases=len(rows),cpu=out['cpu'])))
if __name__=='__main__':main()
