"""Saved exact-range and conditional GL10 bound census; no integrations."""
from pathlib import Path
from collections import Counter
from fractions import Fraction as F
import gzip,json,hashlib,time,math
import mpmath as mp
P=Path(__file__).resolve().parent

def main():
 start=time.process_time();proto=json.loads((P/'protocol.json').read_text());rows=[];shards=[]
 for p in sorted(P.glob('shard_*.json.gz')):
  d=json.loads(gzip.decompress(p.read_bytes()));assert d['protocol_sha256']==hashlib.sha256((P/'protocol.json').read_bytes()).hexdigest();assert [r['index'] for r in d['rows']]==d['indices'][:len(d['rows'])];rows+=d['rows'];shards.append(dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),complete=d['complete']))
 assert [r['index'] for r in rows]==proto['indices'][:len(rows)];assert all(r['mode']!='failure' for r in rows)
 bounded=[r for r in rows if r['mode']=='bounded'];counts=Counter();bits=[0,0]
 for r in rows:
  v=r['exact_inventory'];counts.update(v['operations']);bits=[max(bits[0],v['max_numerator_bits']),max(bits[1],v['max_denominator_bits'])]
 total=sum((F(r['quadrature']['bound_decimal']) for r in bounded),F(0));qualifying=sum(r['quadrature']['qualifies'] for r in bounded)
 sessions=[json.loads(p.read_text()) for p in sorted(P.glob('session_*.json'))];controls=json.loads((P/'controls.json').read_text())
 with mp.workdps(100):
  decimal=lambda f:mp.nstr(mp.mpf(f.numerator)/f.denominator,90)
  precision=[dict(index=r['index'],relative_difference=float(abs(mp.mpf(r['quadrature']['bound_decimal'])-mp.mpf(r['precision100']['bound_decimal']))/mp.mpf(r['precision100']['bound_decimal']))) for r in bounded if 'precision100' in r]
 out=dict(complete=len(rows)==len(proto['indices']) and all(s['complete'] for s in shards),attempted=len(rows),bounded=len(bounded),uncovered_indices=[r['index'] for r in rows if r['mode']!='bounded'],per_cone_target=proto['per_cone_comparison_target'],qualifying=qualifying,loose_indices=[r['index'] for r in bounded if not r['quadrature']['qualifies']],conditional_bound_sum=float(total),conditional_bound_decimal=decimal(total),aggregate_target=proto['aggregate_comparison_target'],aggregate_comparison_pass=bool(len(bounded)==len(proto['indices']) and total<=F(proto['aggregate_comparison_target'])),max_bound=max((r['quadrature']['bound'] for r in bounded),default=0),lower_method_counts=dict(Counter(r['lower_method'] for r in bounded)),reused_prior_certificates=sum('reused_method' in r for r in rows),source_thin_mass=math.fsum(r['source_value'] for r in rows),reference_thin_mass=math.fsum(r['reference'] for r in rows),reference_bound_sum=math.fsum(r['reference_bound'] for r in rows),observed_thin_signed_error=math.fsum(r['observed_difference'] for r in rows),observed_thin_L1_error=math.fsum(abs(r['observed_difference']) for r in rows),exact_operation_inventory=dict(counts),max_numerator_bits=bits[0],max_denominator_bits=bits[1],certificate_cpu=math.fsum(r['certificate_cpu'] for r in rows),bound_cpu=math.fsum(r['quadrature']['bound_cpu']+r.get('precision100',{}).get('bound_cpu',0) for r in bounded),staged_cpu=math.fsum(s['cpu'] for s in sessions),controls_cpu=controls['cpu'],precision_checks=precision,shards=shards,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Real-arithmetic ideal GL10 truncation on stored binary rays in observed thin branch only. MP bound arithmetic not directed; floating source determinant/nodes/weights/kernel and original geometry excluded. Log128 excluded; no fullcost/FP/accuracy certificate.')
 out['aggregation_cpu']=time.process_time()-start;(P/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('shards','exact_operation_inventory','loose_indices','uncovered_indices')},indent=2))
if __name__=='__main__':main()
