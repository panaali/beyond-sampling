"""Independent exact-rational saved-array audit of fixed wide-rule transform."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time,numpy as np
start=time.process_time();H=Path(__file__).parent;P=Path('/workspace/work/recover-small_h2_910/m16/approaches');C=P/'cubic_wide_rule';S=P/'leaf_mass_review';D=P/'plackett_rule_doubling';summary=json.loads((C/'summary.json').read_text());indices=[];data=[[] for _ in range(4)];cost=controls=unchanged=wide=thin=cutoff=0;cpu=0
for b in range(8):
 m=json.loads((C/f'batch_{b}.json').read_text());assert m['complete']
 for path,sha in m['source_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
 with np.load(C/f'batch_{b}.npz') as x,np.load(S/f'error_batch_{b}.npz') as r,np.load(D/f'batch_{b}.npz') as d:
  assert np.array_equal(x['indices'],r['indices']) and np.array_equal(x['indices'],d['indices']);assert np.array_equal(x['thin'],d['thin']) and np.array_equal(x['cutoff'],d['cutoff']);indices.extend(x['indices'].tolist());mask=x['thin']|x['cutoff'];assert np.array_equal(x['values'][mask].view(np.uint64),r['source'][mask].view(np.uint64));unchanged+=int(mask.sum());wide+=int((~mask).sum());thin+=int((x['thin']&~x['cutoff']).sum());cutoff+=int(x['cutoff'].sum())
  for dest,arr in zip(data,[x['values'],r['source'],r['positive'],r['bounds']]):dest.extend(map(lambda v:F(float(v)),arr))
  expected=sum(156+(0 if c else 152*(48 if t else 12)) for t,c in zip(x['thin'],x['cutoff']));assert expected==m['mass_named_ops'];cost+=expected;controls+=m['controls_named_ops'];cpu+=m['cpu']
assert sorted(indices)==list(range(35098));candidate,old,ref,bounds=data
acc=float(1-sum(candidate,F(0)));truth=float(1-sum(ref,F(0)));delta=sum((a-b for a,b in zip(candidate,ref)),F(0));l1=sum((abs(a-b) for a,b in zip(candidate,ref)),F(0));neg=[a for a in candidate if a<0];bound=float(sum(bounds,F(0)))
assert abs(acc-summary['cubic_accuracy'])<2e-16 and abs(truth-summary['reference_accuracy'])<2e-16;assert abs(float(delta)-summary['cubic_errors']['signed_error'])<1e-19 and abs(float(l1)-summary['cubic_errors']['absolute_error_sum'])<1e-18
assert len(neg)==summary['cubic_errors']['negative_count']==2286;assert cost==summary['mass_named_ops']==204957048;assert controls==summary['control_named_ops'];assert bound==summary['conditional_reference_bound']
out=dict(all_indices_once=len(indices),cubic_accuracy=acc,positive_reference_accuracy=truth,error_mass_signed_difference=float(delta),per_cone_L1=float(l1),negative_terms=len(neg),bitwise_unchanged_thin_or_cutoff=unchanged,wide=wide,thin=thin,cutoff=cutoff,inherited_mass_count=cost,conditional_reference_bound=bound,author_recorded_cpu=cpu,review_cpu=time.process_time()-start,scope='Fixed known seed0 geometry and one transform, no quadrature rerun. Wide12/p3 improves original wide12/p2 but retains negative masses; no replacement of frozen doubled-rule candidate or portable/full-cost claim.')
(H/'cubic_rule_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
