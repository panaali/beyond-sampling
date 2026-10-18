import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
import hashlib,itertools,json,time
import numpy as np
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/rank_numeric_envelope');start=time.process_time();protocol=json.loads((P/'protocol.json').read_text());d=json.loads((P/'reflector_controls.json').read_text())
for path,digest in protocol['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
c=protocol['reflector_controls'];expected=set(itertools.product(c['n'],c['incx'],c['alpha'],c['x_patterns']));seen=set();rescales=0
for r in d['rows']:
 key=(r['n'],r['incx'],r['alpha'],r['pattern']);assert key not in seen;seen.add(key);s=r['source'];native=r['native'];a=np.array([s['alpha'],s['tau']]+s['x']);b=np.array([native['alpha'],native['tau']]+native['x']);assert np.array_equal(a.view(np.uint64),b.view(np.uint64));k=s['rescale_iterations'];assert 0<=k<=20;rescales+=bool(k)
 if r['n']<=1:counts={};children={}
 elif r['pattern']=='zero':counts={'comparison':1};children={'dnrm2':1}
 else:
  counts={'arithmetic':5 if k==0 else 6+3*k,'negation':1+bool(k),'sign':1+bool(k),'abs':1+k,'comparison':2+k};children={'dnrm2':1+bool(k),'dlapy2':1+bool(k),'dlamch':2,'dscal':1+k}
 assert s['counts']==counts and s['children']==children,(key,s['counts'],counts)
 for name,v in counts.items():assert v<=d['numeric_exclusive_bound'][name]
assert seen==expected and len(seen)==160 and sum(d['numeric_exclusive_bound'].values())==113
out=dict(cases=160,bitwise_output_and_exact_branch_counts=True,rescaling_cases=rescales,generic_exclusive_bound=113,no_rescale_exclusive_bound=10,review_cpu=time.process_time()-start,scope='Source exclusive numeric rules and saved wrapper/native parity; shared native children are not independently priced or verified. No finite no-rescale census deduction without complete call observation, no full SVD cost.')
Path(__file__).with_name('reflector_saved_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
