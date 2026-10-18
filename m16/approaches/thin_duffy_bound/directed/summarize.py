"""Directed total bound and explicit arithmetic accounting from saved records."""
from pathlib import Path
from collections import Counter
from fractions import Fraction as F
from decimal import Decimal
import gzip,json,hashlib,time
import primitive as A
P=Path(__file__).resolve().parent

def main():
 start=time.process_time();rows=[];shards=[]
 for p in sorted(P.glob('shard_*.json.gz')):
  d=json.loads(gzip.decompress(p.read_bytes()));assert d['protocol_sha256']==hashlib.sha256((P/'protocol.json').read_bytes()).hexdigest();assert [r['index'] for r in d['rows']]==d['indices'][:len(d['rows'])];rows+=d['rows'];shards.append(dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),complete=d['complete']))
 assert [r['index'] for r in rows]==A.PROTOCOL['indices'][:len(rows)];assert all(not r.get('failure') for r in rows)
 m=A.load();D=m.Directed(50);intervals=[m.Interval(Decimal(r['lower_formula_endpoint']),Decimal(r['upper_bound'])) for r in rows];total=D.sum(intervals);exact_upper_sum=sum((F(r['upper_bound']) for r in rows),F(0));assert F(total.hi)>=exact_upper_sum
 sessions=[json.loads(p.read_text()) for p in sorted(P.glob('session_*.json'))];assert sessions
 cold=sessions[0]['cold_counts'];assert all(s['cold_counts']==cold for s in sessions);known=Counter(cold)
 for r in rows:known.update(r['counts']);assert r['qualifies']==(F(r['upper_bound'])<=F(1,10**7*35098))
 known.update(D.counts);complete=len(rows)==17955 and all(s['complete'] for s in shards)
 original=json.loads((P.parent/'rescue/results.json').read_text());old={r['index']:r['quadrature']['bound_decimal'] for p in sorted(P.parent.glob('shard_*.json.gz')) for r in json.loads(gzip.decompress(p.read_bytes()))['rows']}
 old.update({r['index']:r['quadrature']['bound_decimal'] for r in original['rows']});assert all(F(r['upper_bound'])>=F(old[r['index']]) for r in rows)
 out=dict(complete=complete,cases=len(rows),qualified=sum(r['qualifies'] for r in rows),loose_indices=[r['index'] for r in rows if not r['qualifies']],reused_kkt54=sum(r['reused_kkt_rescue'] for r in rows),directed_aggregate_upper=str(total.hi),exact_sum_of_upper_endpoints=str(exact_upper_sum),aggregate_target='1/10000000',aggregate_comparison_pass=bool(complete and F(total.hi)<=F(1,10**7)),prior_non_directed_bound=original['aggregate_bound_sum'],known_directed_category_counts=dict(known),cold_once_counts=cold,final_directed_sum_counts=dict(D.counts),actual_staged_cold_count_multiplier=len(sessions),staged_cpu=sum(s['cpu'] for s in sessions),max_rss_kib=max(s['max_rss_kib'] for s in sessions),shards=shards,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Upper bound on ideal GL10 truncation for fixed stored binary rays under reviewed Decimal semantics; not a lower error bound. Actual estimator floating determinant/nodes/kernel/original geometry/log128 and full model/cost certification excluded.')
 out['aggregation_cpu']=time.process_time()-start;(P/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('shards','exact_sum_of_upper_endpoints')},indent=2))
if __name__=='__main__':main()
