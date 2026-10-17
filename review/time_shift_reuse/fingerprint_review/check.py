"""Independent saved16 audit: determinant minors and direct subset sums."""
from fractions import Fraction as F
from collections import Counter,defaultdict
from pathlib import Path
import hashlib,itertools,json,resource,signal,time
HERE=Path(__file__).resolve().parent
AUTHOR=Path('/workspace/work/recover-small_h2_56/m16/approaches/future_equivalence/identifiability')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def q(v):return F(int(v[0]),int(v[1]))
def det(A):
 n=len(A);out=F(0)
 for p in itertools.permutations(range(n)):
  term=F((-1)**sum(p[i]>p[j] for i in range(n) for j in range(i+1,n)))
  for i,j in enumerate(p):term*=A[i][j]
  out+=term
 return out

def rank(A):
 m=len(A);n=len(A[0])
 for k in range(min(m,n),0,-1):
  for rows in itertools.combinations(range(m),k):
   for cols in itertools.combinations(range(n),k):
    if det([[A[i][j] for j in cols] for i in rows]):return k
 return 0

def stop(*args):raise TimeoutError('19CPU soft cap')
def main():
 resource.setrlimit(resource.RLIMIT_CPU,(19,20));signal.signal(signal.SIGXCPU,stop);dst=HERE/'results.json';assert not dst.exists();start=time.process_time();out=dict(complete=False,rows=[],claim='4685f6ae',author_commit='98b661f7',new_author_trained_mass_calls=0)
 def save():dst.write_text(json.dumps(out,indent=2)+'\n')
 save()
 try:
  manifest=json.loads((HERE/'protocol.json').read_text())
  for p,h in manifest['pins'].items():assert sha(p)==h,p
  p=json.loads((AUTHOR/'protocol.json').read_text());d=json.loads((AUTHOR/'controls.json').read_text());progress=[json.loads(x) for x in (AUTHOR/'progress.jsonl').read_text().splitlines()]
  assert d['complete'] and d['all_pass'] and len(d['rows'])==len(p['cases'])==16
  assert len(progress)==33 and progress[0]['stage']=='attempt' and progress[0]['protocol_sha256']==sha(AUTHOR/'protocol.json')
  assert p['source_gate']!='pending'
  for path,h in p['pins'].items():assert sha(path)==h,path
  kinds=Counter()
  for idx,(case,row) in enumerate(zip(p['cases'],d['rows'],strict=True)):
   out['active']=case['name'];save()
   assert row['name']==case['name'] and row['passed'] is True
   assert progress[1+2*idx]==dict(stage='case_attempt',name=case['name'])
   assert progress[2+2*idx]==dict(stage='case_result',row=row)
   if 'reject' in case:
    expected={'empty':'AssertionError','matrix_shape':'AssertionError','readout_shape':'AssertionError','not_rational':'ValueError'}
    assert row['rejected']==expected[case['name']]
    kinds['rejected']+=1;out['rows'].append(dict(name=case['name'],rejected=row['rejected']));save();continue
   r=row['result'];Wi=list(map(F,case['model'][0]));Wh=[list(map(F,x)) for x in case['model'][1]];Wo=[list(map(F,x)) for x in case['model'][2]];H=len(Wi);K=len(Wo)-1
   assert list(map(q,r['model'][0]))==Wi and [[q(v) for v in x] for x in r['model'][1]]==Wh and [[q(v) for v in x] for x in r['model'][2]]==Wo
   R=[[q(v) for v in x] for x in r['rref']];rk=rank(Wh)
   assert r['rank']==rk and rank(R)==rk and rank(Wh+R)==rk
   piv=[next((j for j,v in enumerate(x) if v),None) for x in R];nonnull=[v for v in piv if v is not None]
   assert nonnull==sorted(set(nonnull))==r['pivots'] and piv[:rk]==nonnull and all(v is None for v in piv[rk:])
   for i,j in enumerate(nonnull):assert all(R[k][j]==F(k==i) for k in range(H))
   assert r['swaps']==[]
   C=[tuple(abs(Wi[i])*(Wo[k+1][i]-Wo[0][i]) for k in range(K)) for i in range(H)]
   sums=[tuple(sum((C[i][k] for i in range(H) if mask&(1<<i)),F(0)) for k in range(K)) for mask in range(1<<H)]
   assert [tuple(map(q,x)) for x in r['fingerprints']]==C and [tuple(map(q,x)) for x in r['subset_sums']]==sums
   groups=defaultdict(list)
   for i,x in enumerate(sums):groups[x].append(i)
   assert sorted(x for x in groups.values() if len(x)>1)==sorted(r['collision_groups'])
   checks=dict(nonzero_input=all(x!=0 for x in Wi),recurrent_invertible=rk==H,subset_injective=len(set(sums))==1<<H)
   assert checks==case['checks']==r['checks'] and r['identifiable'] is all(checks.values())
   # Frozen fixtures: only one nondiagonal matrix, rank-one [[1,2],[2,4]].
   if case['name']=='singular':expected=Counter(div=2,mul=2,sub=2);assert Wh==[[F(1),F(2)],[F(2),F(4)]]
   else:
    assert all(Wh[i][j]==0 for i in range(H) for j in range(H) if i!=j) and all(Wh[i][i]!=0 for i in range(H));expected=Counter(div=H*H)
   assert dict(expected)==r['rank_counts']
   expected.update(abs=H,sub=H*K,mul=H*K,add=((1<<H)-1)*K)
   assert dict(expected)==r['counts']
   kind='sufficient_positive' if r['identifiable'] else 'inconclusive_predicate_failure';kinds[kind]+=1
   out['rows'].append(dict(name=case['name'],rank=rk,subsets=len(sums),distinct_sums=len(set(sums)),checks=checks,named_counts=dict(expected),status=kind));save()
  assert kinds==dict(sufficient_positive=7,inconclusive_predicate_failure=5,rejected=4)
  out.update(complete=True,active=None,counts=dict(kinds),progress_records=len(progress),pins=manifest['pins'],author_reported_cpu=d['cpu_seconds'],scope='Saved exact rank/subset/condition/count/progress audit, no author imports or numerical rerun. Does not independently replay observed-bit maxima or prove all RREF branches; frozen fixtures have no row swaps. Failed sufficient predicates are inconclusive. No trained/state-domain/label/mass/runtime claim.')
 except BaseException as e:out['failure']=repr(e);raise
 finally:out['cpu_seconds']=time.process_time()-start;save()
 print(json.dumps(dict(complete=out['complete'],rows=len(out['rows']),cpu=out['cpu_seconds'])))
if __name__=='__main__':main()
