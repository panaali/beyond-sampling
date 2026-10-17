"""Replay saved contributions only; exhaustive index checks and analytic chambers."""
import hashlib,json,math,time
from fractions import Fraction
from pathlib import Path
import numpy as np
START=time.process_time();HERE=Path(__file__).resolve().parent
capture=json.loads((HERE/'capture.json').read_text());archive=np.load(HERE/'captured.npz');parents=archive['parent_ids'];N=len(parents)
rows=[];meta=[]
for b in range(8):
 d=json.loads((HERE/f'full_batch_{b}.json').read_text());a=np.load(HERE/f'full_batch_{b}.npz');assert d['complete'];meta.append(d)
 for name,digest in d['source_hashes'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
 assert np.all((parents[a['indices']]>=b*12)&(parents[a['indices']]<(b+1)*12))
 rows.extend(zip(a['indices'].tolist(),a['source'].tolist(),a['positive'].tolist(),a['bounds'].tolist(),a['orders'].tolist(),a['reused'].tolist()))
rows.sort();assert [r[0] for r in rows]==list(range(N));assert all(math.isfinite(x) for r in rows for x in r[1:4]);assert all(r[3]>=0 for r in rows)
pairs=[]
for i in range(0,96,2):
 p,q=capture['parents'][i:i+2];assert p['perm']==q['perm'] and p['sigma']==q['sigma'];rank=p['perm'].index(0)+1
 count=sum(math.comb(4,k) for k in range(rank,5));count=count if p['sigma']>0 else 16-count;exact=Fraction(count,384)
 chosen=[r for r in rows if parents[r[0]] in (i,i+1)];old=math.fsum(r[1] for r in chosen);positive=math.fsum(r[2] for r in chosen);bound=math.fsum(r[3] for r in chosen)
 pairs.append(dict(parents=[i,i+1],permutation=p['perm'],sigma=p['sigma'],analytic=str(exact),simplices=len(chosen),source=old,positive=positive,source_error=old-float(exact),positive_error=positive-float(exact),conditional_bound=bound))
assert sum(Fraction(p['analytic']) for p in pairs)==1
out=dict(complete=True,gate_simplices=N,gate_leaves=capture['leaf_count'],all_indices_once=True,source_sum=math.fsum(r[1] for r in rows),positive_sum=math.fsum(r[2] for r in rows),conditional_integration_bound=math.fsum(r[3] for r in rows),source_negative_count=sum(r[1]<0 for r in rows),source_negative_sum=math.fsum(r[1] for r in rows if r[1]<0),positive_negative_count=sum(r[2]<0 for r in rows),bound_misses=sum(r[3]>1e-9/N for r in rows),reused_selected_values=sum(r[5] for r in rows),order_counts={str(n):sum(r[4]==n for r in rows) for n in (20,32,48)},full_replay_cpu=sum(d['cumulative_cpu'] for d in meta),prior_capture_cpu=capture['cpu_seconds_including_imports'],prior_selected_successful_cpu=5.303671444,prior_failed_selected_cpu='unknown; additional failed864-simplex pass, never zero',full_replay_partial_source_named_ops=sum(d['cumulative_source_named_ops'] for d in meta),max_absolute_chamber_positive_error=max(abs(p['positive_error']) for p in pairs),pairs=pairs,scope='Integration-only conditional bounds on captured floating geometry; no accuracy or portable cost claim; no proof against compensating topology defects.')
out['source_excess']=out['source_sum']-1;out['positive_error']=out['positive_sum']-1;out['summary_cpu']=time.process_time()-START
(HERE/'full_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='pairs'},indent=2))
