"""Saved-array output-error totals, scope and pre-existing MC diagnostics."""
import hashlib,json,math,time
from pathlib import Path
import numpy as np
START=time.process_time();HERE=Path(__file__).resolve().parent
capture=json.loads((HERE/'error_capture.json').read_text());assert capture['complete'];a=np.load(HERE/'error_captured.npz');N=len(a['rays']);rows=[];meta=[]
for b in range(8):
 d=json.loads((HERE/f'error_batch_{b}.json').read_text());v=np.load(HERE/f'error_batch_{b}.npz');assert d['complete'];meta.append(d)
 for p,h in d['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 assert np.all((a['parent_ids'][v['indices']]>=12*b)&(a['parent_ids'][v['indices']]<12*(b+1)))
 rows.extend(zip(v['indices'].tolist(),v['source'].tolist(),v['positive'].tolist(),v['bounds'].tolist(),v['orders'].tolist()))
rows.sort();assert [r[0] for r in rows]==list(range(N));assert all(math.isfinite(x) for r in rows for x in r[1:4]);assert all(r[3]>=0 for r in rows)
source=math.fsum(r[1] for r in rows);positive=math.fsum(r[2] for r in rows);bound=math.fsum(r[3] for r in rows);accuracy=1-positive
reference=Path('/workspace/work/admin-t121-rank-mass/m16/approaches/final3/exact_H8_T4.json');prior=json.loads(reference.read_text())['zoo0']['acc']
assert abs(1-source-prior)<2e-15
comparisons=[]
for name,p,n,precision in [('numpy recorded4M',.97606325,4000000,0.),('torch README rounded10M',.97617,10000000,.000005)]:
 se=math.sqrt(p*(1-p)/n);comparisons.append(dict(source=name,probability=p,samples=n,rounding_halfwidth=precision,plug_in_standard_error=se,reference_minus_mc=accuracy-p,z_score=(accuracy-p)/se,diagnostic_only=True))
out=dict(complete=True,error_simplices=N,gate_leaves=capture['counts']['leaves'],all_indices_once=True,source_error_mass=source,positive_error_mass=positive,source_accuracy=1-source,positive_accuracy=accuracy,source_minus_positive_error=source-positive,conditional_integration_bound=bound,source_negative_count=sum(r[1]<0 for r in rows),source_negative_sum=math.fsum(r[1] for r in rows if r[1]<0),positive_negative_count=sum(r[2]<0 for r in rows),bound_misses=sum(r[3]>1e-9/N for r in rows),order_counts={str(n):sum(r[4]==n for r in rows) for n in (20,32,48)},replay_cpu=sum(m['cumulative_cpu'] for m in meta),capture_cpu=capture['cpu_seconds_including_imports'],source_partial_named_mass_ops=sum(m['cumulative_source_named_ops'] for m in meta),unchanged_source_matches_prior=True,prior_source_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),mc_diagnostics=comparisons,scope='Integration bound conditional on captured floating geometry and arithmetic; no full roundoff/topology certificate, portable-cost claim or estimator competitiveness. MC comparisons diagnostic only; torch value rounded in existing README.')
out['aggregation_cpu']=time.process_time()-START;(HERE/'error_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
