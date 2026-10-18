import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import itertools,json,hashlib,time,numpy as np
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/rank_numeric_envelope');start=time.process_time();p=json.loads((P/'protocol.json').read_text());d=json.loads((P/'controls.json').read_text());assert d['complete'] and len(d['rows'])==64
expected=set(itertools.product(p['controls']['active_lengths'],p['controls']['i0'],p['controls']['pp'],p['controls']['tau'],p['controls']['ieee']));seen=set();maxima={}
for r in d['rows']:
 key=(r['length'],r['i0'],r['pp'],r['tau'],r['ieee']);assert key in expected and key not in seen;seen.add(key);a,b=r['source'],r['native'];L=r['length'];bound=dict(arithmetic=5*L+1 if r['ieee'] else 6*L-2,negation=1,comparison=3*L-5 if r['ieee'] else 4*L-6)
 assert r['bound']==bound and all(a['counts'].get(k,0)<=v for k,v in bound.items());fields=['tau','dmin','dmin1','dmin2','dn','dnm1','dnm2'];x=np.array(a['z']+[a[k] for k in fields]);y=np.array(b['z']+[b[k] for k in fields]);assert np.array_equal(x.view(np.uint64),y.view(np.uint64));maxima[L,r['ieee']]=max(maxima.get((L,r['ieee']),0),sum(a['counts'].values()))
assert seen==expected and maxima=={(3,False):23,(3,True):21,(4,False):33,(4,True):29}
path=next(Path(x) for x in p['pins'] if x.endswith('/rank_runtime_census/summary.json'));assert hashlib.sha256(path.read_bytes()).hexdigest()==p['pins'][str(path)];census=json.loads(path.read_text());modes={}
for name,mode in census['modes'].items():
 rows=[r for r in mode['rows'] if r['kernel']=='dlasq5'];counts=Counter()
 for r in rows:
  L=r['m'];assert L in [3,4] and r['n'] in [0,1] and not r['query'];counts.update({k:v*r['calls'] for k,v in dict(arithmetic=6*L-2,negation=1,comparison=4*L-6).items()})
 assert sum(r['calls'] for r in rows)==mode['kernel_calls']['dlasq5'];modes[name]=dict(calls=sum(r['calls'] for r in rows),exclusive_numeric_envelope=dict(counts),total=sum(counts.values()))
out=dict(controls=64,maxima={str(k):v for k,v in maxima.items()},actual_finite_census_subtotals=modes,review_cpu=time.process_time()-start,scope='Exclusive pinned Q5 source arithmetic/negation/numeric comparison upper subtotal; observed calls only. Other kernels/index/control/memory remain unpriced; not a universal SVD bound or full hardware cost.')
Path(__file__).with_name('q5_envelope_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
