"""Independent fixed rank/minor, trace and Leibniz-record review."""
from check_t208_implementation_saved import rank,determinant,product
from check_domain_sufficient_saved import unpack,sha,H
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import itertools,json,time
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/future_equivalence/domain_sufficient/rank_witness')
def bits(values):return [max((abs(x.numerator).bit_length() for x in values),default=0),max((x.denominator.bit_length() for x in values),default=0)]
def main():
 start=time.process_time();raw=json.loads((P/'controls.json').read_text());d=unpack(raw);p=json.loads((P/'protocol.json').read_text());events=[json.loads(x) for x in (P/'controls_progress.jsonl').read_text().splitlines()]
 assert d['complete'] and d['all_pass'] and d['cases']==12 and d['protocol_sha256']==sha(P/'protocol.json')
 for f,h in p['pins'].items():assert sha(f)==h
 assert len(d['rows'])==8 and len(d['guards'])==4
 assert [r['row'] for r in events if r['stage']=='result']==raw['rows']+raw['guards'] and Counter(r['stage'] for r in events)==dict(phase_attempt=1,attempt=12,result=12,complete=1)
 ranks=[]
 for row in d['rows']:
  R=[[F(x) for x in v] for v in row['rays']];ans=row['result'];r=rank(R);assert r==row['expected_rank']==ans['rank'] and row['passed'];ranks.append(r)
  selected=[]
  for i in range(len(R)):
   if rank([R[j] for j in selected]+[R[i]])>len(selected):selected.append(i)
   if len(selected)==4:break
  assert selected==ans['selected_indices'];limit=selected[-1]+1 if r==4 else len(R)
  assert ans['total_ray_rows']==len(R) and ans['considered_ray_rows']==len(ans['trace'])==limit and ans['rational_decodes']==4*len(R)
  basis=[];pivots=[];counts=Counter();values=[]
  for i,record in enumerate(ans['trace']):
   z=R[i][:]
   for pivot,b in zip(pivots,basis,strict=True):
    c=z[pivot]
    if c:
     products=[c*x for x in b];values+=products;z=[x-y for x,y in zip(z,products,strict=True)];values+=z;counts['mul']+=4;counts['sub']+=4
   pivot=next((j for j,x in enumerate(z) if x),None);assert record['index']==i and record['residual']==z and record['pivot']==pivot
   if pivot is not None:
    b=[x/z[pivot] for x in z];assert record['normalized']==b;values+=b;counts['div']+=4;basis.append(b);pivots.append(pivot)
  assert ans['pivots']==pivots and ans['elimination_counts']==dict(counts) and ans['elimination_observed_bits']==bits(values)
  if r<4:assert ans['determinant'] is None and ans['status']=='unknown_rank_deficient'
  else:
   a=[R[j] for j in selected];det=ans['determinant'];assert det['value']==row['expected_determinant']==determinant(a)!=0 and ans['status']=='full_rank_witness'
   terms=[];values=[];total=F(0)
   for perm in itertools.permutations(range(4)):
    v=a[0][perm[0]]
    for i in range(1,4):v*=a[i][perm[i]];values.append(v)
    odd=bool(sum(perm[i]>perm[j] for i in range(4) for j in range(i+1,4))%2);total+=-v if odd else v;values.append(total);terms.append(dict(permutation=list(perm),odd=odd,product=v))
   assert det['terms']==terms and det['counts']==dict(mul=72,add=12,sub=12) and det['observed_bits']==bits(values) and total==det['value']
 assert d['guards']==[dict(name=n,error=e,passed=True) for n,e in [('empty','ValueError'),('wrong_width','ValueError'),('ragged','ValueError'),('nonfinite','OverflowError')]]
 out=dict(complete=True,source_result_commit='52cd44c4',pins={n:sha(P/n) for n in ['controls.json','controls_progress.jsonl','protocol.json']},checker_sha256=sha(Path(__file__)),cases=8,guards=4,ranks=ranks,exact_determinants=5,review_cpu=time.process_time()-start,author_cpu=d['cpu_seconds'],new_author_trained_geometry_calls=0,scope='Fixed synthetic saved controls only; determinant-minor rank, selected original rays, elimination trace,24determinant terms,named counts and observed output bits independently checked. Guard outcomes checked without author invalid-path replay. No trained application, feasibility, label or cost conclusion.')
 (H/'rank_witness_saved_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(complete=True,review_cpu=out['review_cpu'])))
if __name__=='__main__':main()
