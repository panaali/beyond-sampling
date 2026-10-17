"""Fixed-transform totals, per-cone errors and unchanged thin-branch control."""
import hashlib,json,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;SOURCE=HERE.parent/'leaf_mass_review';DOUBLE=HERE.parent/'plackett_rule_doubling';rows=[];meta=[]
for b in range(8):
 d=json.loads((HERE/f'batch_{b}.json').read_text());assert d['complete'];meta.append(d);a=np.load(HERE/f'batch_{b}.npz');ref=np.load(SOURCE/f'error_batch_{b}.npz');double=np.load(DOUBLE/f'batch_{b}.npz');assert np.array_equal(a['indices'],ref['indices']) and np.array_equal(a['indices'],double['indices']);assert np.array_equal(a['thin'],double['thin']) and np.array_equal(a['cutoff'],double['cutoff'])
 for p,h in d['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 rows.extend(zip(a['indices'].tolist(),a['values'].tolist(),ref['source'].tolist(),ref['positive'].tolist(),ref['bounds'].tolist(),a['thin'].tolist(),a['cutoff'].tolist(),double['values'].tolist()))
rows.sort();assert [r[0] for r in rows]==list(range(35098));assert all(r[1]==r[2] for r in rows if r[5] or r[6])
def errors(col,subset):
 e=[r[col]-r[3] for r in subset];absolute=[abs(v) for v in e];return dict(count=len(subset),signed_error=math.fsum(e),absolute_error_sum=math.fsum(absolute),max_absolute_error=max(absolute),absolute_error_quantiles={str(q):float(np.quantile(absolute,q)) for q in (.5,.9,.99,.999)},negative_count=sum(r[col]<0 for r in subset),negative_sum=math.fsum(r[col] for r in subset if r[col]<0))
oldops=sum(json.loads((SOURCE/f'error_batch_{b}.json').read_text())['cumulative_source_named_ops'] for b in range(8));newops=sum(d['mass_named_ops'] for d in meta);expected=sum(156+(0 if r[6] else 152*(48 if r[5] else 12)) for r in rows);assert newops==oldops==expected
out=dict(complete=True,count=len(rows),all_indices_once=True,original_accuracy=1-math.fsum(r[2] for r in rows),cubic_accuracy=1-math.fsum(r[1] for r in rows),doubled_accuracy=1-math.fsum(r[7] for r in rows),reference_accuracy=1-math.fsum(r[3] for r in rows),conditional_reference_bound=math.fsum(r[4] for r in rows),original_errors=errors(2,rows),cubic_errors=errors(1,rows),doubled_errors=errors(7,rows),wide_original=errors(2,[r for r in rows if not r[5] and not r[6]]),wide_cubic=errors(1,[r for r in rows if not r[5] and not r[6]]),thin_and_cutoff_unchanged_bitwise=True,mass_named_ops=newops,original_full_partial_named_ops=240607902,control_named_ops=sum(d['controls_named_ops'] for d in meta),cpu=sum(d['cpu'] for d in meta),scope='Exploratory one-transform comparison on known seed0 corpus, not holdout/generalization. Same nodecounter, not complete portable/hardwarecost. Frozen24/96 T149candidate unchanged.')
(HERE/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
