"""Independent saved trained-weight check using integer determinants/minors."""
from fractions import Fraction as F
from collections import Counter
from pathlib import Path
import hashlib,json,math,resource,signal,time
HERE=Path(__file__).resolve().parent
AUTHOR=Path('/workspace/work/recover-small_h2_56/m16/approaches/future_equivalence/identifiability')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def q(v):return F(int(v[0]),int(v[1]))
def det(A):
 n=len(A)
 if not n:return F(1)
 D=math.lcm(*(v.denominator for row in A for v in row));B=[[int(v*D) for v in row] for row in A];previous=1;sign=1
 for k in range(n-1):
  pivot=next((i for i in range(k,n) if B[i][k]),None)
  if pivot is None:return F(0)
  if pivot!=k:B[k],B[pivot]=B[pivot],B[k];sign=-sign
  p=B[k][k]
  for i in range(k+1,n):
   for j in range(k+1,n):
    value=p*B[i][j]-B[i][k]*B[k][j];assert value%previous==0;B[i][j]=value//previous
   B[i][k]=0
  previous=p
 return F(sign*B[-1][-1],D**n)
def stop(*args):raise TimeoutError('19CPU soft cap')
def main():
 resource.setrlimit(resource.RLIMIT_CPU,(19,20));signal.signal(signal.SIGXCPU,stop);dst=HERE/'trained_results.json';assert not dst.exists();start=time.process_time();out=dict(complete=False,rows=[],claim='36c46b08',new_author_model_geometry_mass_calls=0)
 def save():dst.write_text(json.dumps(out,indent=2)+'\n')
 save()
 try:
  manifest=json.loads((HERE/'trained_protocol.json').read_text())
  for path,h in manifest['pins'].items():assert sha(path)==h,path
  p=json.loads((AUTHOR/'trained_protocol.json').read_text());d=json.loads((AUTHOR/'trained.json').read_text());progress=[json.loads(s) for s in (AUTHOR/'trained_progress.jsonl').read_text().splitlines()]
  assert d['complete'] and len(d['rows'])==3 and len(progress)==8 and d['protocol_sha256']==sha(AUTHOR/'trained_protocol.json')
  assert progress[0]==dict(stage='attempt',protocol_sha256=d['protocol_sha256'])
  for path,h in p['pins'].items():assert sha(path)==h,path
  for i,(spec,row) in enumerate(zip(p['models'],d['rows'],strict=True)):
   out['active']=spec['name'];save();assert row['name']==spec['name'] and row['checkpoint_sha256']==spec['checkpoint_sha256']
   assert progress[2*i+1]==dict(stage='model_attempt',name=spec['name']) and progress[2*i+2]==dict(stage='model_result',row=row)
   archive=json.loads(Path(spec['path']).read_text());assert archive['complete']
   if spec['kind']=='capture':
    assert archive['checkpoint_sha256']==spec['checkpoint_sha256'];m=archive['model'];Wi=list(map(q,m[0]));Wh=[list(map(q,x)) for x in m[1]];Wo=[list(map(q,x)) for x in m[2]]
   else:
    rec=next(x for x in archive['models'] if x['name']==spec['name']);assert rec['checkpoint_sha256']==spec['checkpoint_sha256'];m=rec['model'];D=m['D'];Wi=[F(v,D) for v in m['wi']];Wh=[[F(v,D) for v in x] for x in m['wh']];Wo=[[F(v,D) for v in x] for x in m['wo']]
   r=row['answer'];H=len(Wi);K=len(Wo)-1;assert H==spec['H'] and K+1==spec['T']
   assert list(map(q,r['model'][0]))==Wi and [list(map(q,x)) for x in r['model'][1]]==Wh and [list(map(q,x)) for x in r['model'][2]]==Wo
   determinant=det(Wh);assert determinant!=0 and r['rank']==H and r['pivots']==list(range(H)) and not r['swaps']
   assert [list(map(q,x)) for x in r['rref']]==[[F(i==j) for j in range(H)] for i in range(H)]
   # Independent Schur/Cramer minors verify no zero elimination coefficient:
   # earlier pivot rows use B_k^-1*c; later rows use bordered determinants.
   minors=0
   for k in range(H):
    B=[x[:k] for x in Wh[:k]];assert det(B)!=0 and det([x[:k+1] for x in Wh[:k+1]])!=0
    for j in range(H):
     if j==k:continue
     if j<k:
      A=[x[:] for x in B]
      for ii in range(k):A[ii][j]=Wh[ii][k]
     else:A=[x[:k+1] for x in Wh[:k]]+[Wh[j][:k+1]]
     assert det(A)!=0;minors+=1
   rankcounts=dict(div=H*H,mul=H*H*(H-1),sub=H*H*(H-1));assert r['rank_counts']==rankcounts
   C=[tuple(abs(Wi[j])*(Wo[k+1][j]-Wo[0][j]) for k in range(K)) for j in range(H)]
   sums=[tuple(sum((C[j][k] for j in range(H) if mask&(1<<j)),F(0)) for k in range(K)) for mask in range(1<<H)]
   assert [tuple(map(q,x)) for x in r['fingerprints']]==C and [tuple(map(q,x)) for x in r['subset_sums']]==sums
   assert len(set(sums))==len(sums)==spec['subsets'] and not r['collision_groups']
   checks=dict(nonzero_input=all(Wi),recurrent_invertible=True,subset_injective=True);assert r['checks']==checks and r['identifiable'] is True
   expected=Counter(rankcounts);expected.update(abs=H,sub=H*K,mul=H*K,add=((1<<H)-1)*K);assert dict(expected)==r['counts']
   decodes=H+H*H+(K+1)*H;assert row['rational_weight_decodes']==decodes
   out['rows'].append(dict(name=spec['name'],rank=H,determinant=[str(determinant.numerator),str(determinant.denominator)],nonzero_elimination_minors=minors,subsets=len(sums),checks=checks,named_counts=dict(expected),weight_decodes=decodes));save()
  assert progress[-1]==dict(stage='complete',conditions=[r['answer']['checks'] for r in d['rows']])
  assert sum(r['subsets'] for r in out['rows'])==288 and sum(r['weight_decodes'] for r in out['rows'])==172
  out.update(complete=True,active=None,pins=manifest['pins'],author_cpu=d['cpu_seconds'],scope='Three exact stored models only. Independent integer Bareiss/Cramer-minor rank and count contracts, direct exhaustive fingerprints. No author imports or predicate/model rerun. Observed-bit ledger unverified; global one-step margin/projective theorem only, not label/domain/approximate/horizon equivalence or runtime/MSE.')
 except BaseException as e:out['failure']=repr(e);raise
 finally:out['cpu_seconds']=time.process_time()-start;save()
 print(json.dumps(dict(complete=out['complete'],rows=len(out['rows']),cpu=out['cpu_seconds'])))
if __name__=='__main__':main()
