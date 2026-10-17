"""Same frozen panel, signed and absolute errors and named-cost identities."""
import hashlib,json,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;SOURCE=HERE.parent/'leaf_mass_review';rows=[];metadata=[]
for b in range(8):
 d=json.loads((HERE/f'batch_{b}.json').read_text());assert d['complete'];metadata.append(d);a=np.load(HERE/f'batch_{b}.npz');ref=np.load(SOURCE/f'error_batch_{b}.npz');assert np.array_equal(a['indices'],ref['indices'])
 for p,h in d['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 rows.extend(zip(a['indices'].tolist(),a['values'].tolist(),ref['source'].tolist(),ref['positive'].tolist(),ref['bounds'].tolist(),a['thin'].tolist(),a['cutoff'].tolist()))
rows.sort();N=len(rows);assert [r[0] for r in rows]==list(range(35098));candidate=math.fsum(r[1] for r in rows);old=math.fsum(r[2] for r in rows);reference=math.fsum(r[3] for r in rows);bound=math.fsum(r[4] for r in rows)
def errors(col):
 e=[r[col]-r[3] for r in rows];a=np.abs(e);return dict(signed_error=math.fsum(e),absolute_error_sum=math.fsum(a),max_absolute_error=float(a.max()),absolute_error_quantiles={str(q):float(np.quantile(a,q)) for q in (.5,.9,.99,.999)},negative_mass_count=sum(r[col]<0 for r in rows),negative_mass_sum=math.fsum(r[col] for r in rows if r[col]<0),per_cone_within_conditional_reference_bound=sum(abs(r[col]-r[3])<=r[4] for r in rows))
oldcounter=sum(json.loads((SOURCE/f'error_batch_{b}.json').read_text())['cumulative_source_named_ops'] for b in range(8));newcounter=sum(d['mass_named_ops'] for d in metadata);expected=sum(156+(0 if r[6] else 152*(96 if r[5] else 24)) for r in rows);assert newcounter==expected==2*oldcounter-156*N
out=dict(complete=True,count=N,all_indices_once=True,source_error_mass=old,candidate_error_mass=candidate,reference_error_mass=reference,candidate_accuracy=1-candidate,reference_accuracy=1-reference,conditional_reference_integration_bound=bound,original_errors=errors(2),candidate_errors=errors(1),cutoff_count=sum(r[6] for r in rows),cutoff_positive_reference_mass=math.fsum(r[3] for r in rows if r[6]),thin_count=sum(r[5] for r in rows),source_mass_named_ops=oldcounter,candidate_mass_named_ops=newcounter,incremental_named_ops=newcounter-oldcounter,hypothetical_full_named_count=240607902+newcounter-oldcounter,control_named_ops=sum(d['controls_named_ops'] for d in metadata),cpu=sum(d['cpu'] for d in metadata),scope='One fixed checkpoint/rule doubling; inherited counter partial, no portable cost/sampling-win/rung claim. Reference conditional on captured floating geometry. Aggregate cancellation reported explicitly.')
(HERE/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
