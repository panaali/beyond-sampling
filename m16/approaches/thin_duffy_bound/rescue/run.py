"""Fixed54 exact KKT range rescue; ideal GL10 quadrature and all masses unchanged."""
import time
START=time.process_time()
from pathlib import Path
import sys,importlib.util,json,hashlib,gzip,resource
from fractions import Fraction as F
from collections import Counter
import numpy as np
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('thin_prior_primitive',P.parent/'primitive.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
PROTOCOL=json.loads((P/'protocol.json').read_text())

def main():
 resource.setrlimit(resource.RLIMIT_CPU,(20,22))
 for f,h in PROTOCOL['pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
 tracker,radial=A.load();rays=np.load(A.PROTOCOL['rays'])['rays'];rows={r['index']:r for p in sorted(P.parent.glob('shard_*.json.gz')) for r in json.loads(gzip.decompress(p.read_bytes()))['rows']};assert len(rows)==17955
 prior=json.loads((P.parent/'summary.json').read_text());assert prior['complete'];ids=PROTOCOL['indices'];assert ids==prior['loose_indices'] and len(ids)==54
 out=dict(complete=False,rows=[],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((P/'protocol.json').read_bytes()).hexdigest())
 def save():
  out.update(cpu=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);(P/'results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 for i in ids:
  if time.process_time()-START>18:out['stop_reason']='18CPU soft cap';save();return
  old=rows[i];assert 'reused_method' not in old
  try:
   tracker.Tracked.current=tracker.Tracker();begin=time.process_time();cert=radial.bounds(rays[i]);construction_cpu=time.process_time()-begin;construction=tracker.Tracked.current.report()
   assert F(cert['a'])>=F(old['a']) and F(cert['b'])==F(old['b']) and F(cert['det'])==F(old['det'])
   checked=A.certificate(rays[i],tracker,radial,dict(method='exact_kkt',certificate=cert));assert checked['mode']=='bounded' and F(checked['a'])==F(cert['a'])
   result=A.bound(checked);precision=A.bound(checked,100);assert F(result['bound_decimal'])<=F(old['quadrature']['bound_decimal'])
   row=dict(index=i,mode='bounded',prior_bound=old['quadrature'],prior_a=old['a'],certificate=cert,construction_inventory=construction,construction_cpu=construction_cpu,recheck=checked,quadrature=result,precision100=precision)
  except Exception as exc:row=dict(index=i,mode='failure',reason=repr(exc))
  out['rows'].append(row);save()
  if row['mode']=='failure':out['stop_reason']='Explicit failure';save();return
 allbounds={i:F(r['quadrature']['bound_decimal']) for i,r in rows.items()}
 for r in out['rows']:allbounds[r['index']]=F(r['quadrature']['bound_decimal'])
 total=sum(allbounds.values(),F(0));out.update(complete=True,all_cone_count=len(allbounds),unchanged_cones=len(rows)-len(ids),new54_bound_sum=float(sum((F(r['quadrature']['bound_decimal']) for r in out['rows']),F(0))),aggregate_bound_sum=float(total),aggregate_bound_exact_decimal_sum=str(total),aggregate_comparison_pass=total<=F(1,10**7),remaining_loose_indices=[i for i,b in allbounds.items() if b>F(1,10**7*35098)],mass_values_unchanged=True,no_mass_integrations=True,full_floating_certificate=False)
 counts=Counter();reviewcounts=Counter()
 for r in out['rows']:counts.update(r['construction_inventory']['operations']);reviewcounts.update(r['recheck']['exact_inventory']['operations'])
 out.update(construction_inventory=dict(counts),validation_inventory=dict(reviewcounts),construction_cpu=sum(r['construction_cpu'] for r in out['rows']),validation_cpu=sum(r['recheck']['certificate_cpu'] for r in out['rows']),mp_bound_cpu=sum(r['quadrature']['bound_cpu']+r['precision100']['bound_cpu'] for r in out['rows']),max_construction_numerator_bits=max(r['construction_inventory']['max_numerator_bits'] for r in out['rows']),max_construction_denominator_bits=max(r['construction_inventory']['max_denominator_bits'] for r in out['rows']));save();print(json.dumps({k:v for k,v in out.items() if k not in ('rows','construction_inventory','validation_inventory','aggregate_bound_exact_decimal_sum')},indent=2))
if __name__=='__main__':main()
