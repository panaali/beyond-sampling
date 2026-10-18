"""Independent saved T229 control certificates; never imports author code."""
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations
import hashlib,json,time
BASE=Path('/workspace/work/recover-small_h2_34/review/weight_automorphism')
OUT=Path(__file__).with_name('t229_controls_saved_review.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def dec(x):return F(x['rational']) if isinstance(x,dict) and 'rational' in x else x
def main():
 assert not OUT.exists()
 t=time.process_time(); r=read(BASE/'controls_result.json');p=read(BASE/'control_protocol.json');g=read(BASE/'control_gate.json')
 pins={str(BASE/n):sha(BASE/n) for n in ['controls_result.json','controls_progress.jsonl','control_protocol.json','control_gate.json','predicate.py','controls.py','PROOF.md']}
 assert r['complete'] is True and r['pass_all'] is True and r['target_model_calls']==0
 assert r['protocol_sha256']==sha(BASE/'control_protocol.json')==g['protocol_sha256'] and r['gate_sha256']==sha(BASE/'control_gate.json')
 for n,h in p['pins'].items():assert sha(BASE/n)==h
 for n,h in g['pins'].items():assert sha(Path(n))==h
 rv=read(Path(g['review']['path']));assert rv['complete'] is True and rv['pass_all'] is True
 assert g['pins'][g['review']['path']]==g['review']['sha256'] and g['review']['commit'] in g['published_commits']
 valid=[('identity',[[1,0],[0,1]],True),('positive_parallel',[[1,2],[2,4]],False),('negative_parallel',[[1,-2],[2,-4]],True),('zero_column',[[1,0],[2,0]],False),('all_zero',[[0,0],[0,0]],False),('single_nonzero',[[-3]],True),('single_zero',[[0]],False),('cycle_equal_readout',[[1,1,1]],False),('near_parallel_dyadic',[[1,1],[1,1+F(1,2**80)]],True),('three_distinct',[[1,0,1],[0,1,1]],True),('positive_gauge_columns',[[F(1,2),0,3],[0,2,3]],True),('invertible_row_transform',[[1,1],[1,-1]],True)]
 invalid=[('empty',[],'ValueError'),('zero_width',[[]],'ValueError'),('ragged',[[1],[1,2]],'ValueError'),('scalar_row',[1],'TypeError'),('float_coefficient',[[1.0]],'TypeError'),('nonfinite',[[{'float':'inf'}]],'TypeError'),('text_coefficient',[['1/2']],'TypeError'),('bool_coefficient',[[True]],'TypeError')]
 assert len(r['rows'])==20
 minors=divisions=pairs=0
 for row,(name,mat,forced) in zip(r['rows'][:12],valid,strict=True):
  assert row['name']==name and row['pass'] is True
  assert [[dec(x) for x in rr] for rr in row['input']]==mat
  a=row['result']; cols=list(zip(*mat)); h=len(cols); d=len(mat)
  assert a['rows']==d and a['width']==h and [[F(x) for x in c] for c in a['columns']]==[list(c) for c in cols]
  zeros=[j for j,c in enumerate(cols) if all(x==0 for x in c)]
  assert a['zero_columns']==zeros
  canon=a['positive_canonical_columns']
  for j,c in enumerate(cols):
   if j in zeros:assert canon[j] is None
   else:
    first=next(x for x in c if x);v=list(map(F,canon[j]));assert next(x for x in v if x) in (-1,1)
    assert all(vv*abs(first)==x for vv,x in zip(v,c,strict=True))
  groups=[];seen=set();expected_minor=0
  for j,c in enumerate(cols):
   if j in zeros or j in seen:continue
   pivot=next(k for k,x in enumerate(c) if x)
   group=[k for k,b in enumerate(cols) if k not in zeros and b[pivot]*c[pivot]>0 and all(b[z]*c[pivot]==c[z]*b[pivot] for z in range(d))]
   seen.update(group)
   if len(group)>1:groups.append(group)
  assert a['positive_proportional_classes']==groups
  assert forced==(not zeros and not groups)
  status='identity_forced_by_readout' if forced else 'inconclusive'
  assert a['status']==row['expected']==status
  certs=a['pair_certificates'];assert len(certs)==h*(h-1)//2
  for cert,(i,j) in zip(certs,combinations(range(h),2),strict=True):
   pairs+=1;assert cert['pair']==[i,j];u,v=cols[i],cols[j]
   if i in zeros or j in zeros:assert cert['kind']=='zero_column';continue
   pos=any(i in z and j in z for z in groups)
   if pos:
    q=F(cert['ratio_j_over_i']);assert cert['kind']=='positive_proportional' and q>0 and all(y==q*x for x,y in zip(u,v,strict=True));continue
   nonzero=[]
   for x,y in combinations(range(d),2):
    determinant=u[x]*v[y]-u[y]*v[x];expected_minor+=1
    if determinant:nonzero=[x,y,determinant];break
   if nonzero:
    assert cert['kind']=='nonzero_minor' and cert['rows']==nonzero[:2] and F(cert['determinant'])==nonzero[2]
   else:
    q=F(cert['ratio_j_over_i']);assert cert['kind']=='negative_proportional' and q<0 and all(y==q*x for x,y in zip(u,v,strict=True))
  assert a['partial_setup_counts']=={'normalization_divisions':d*(h-len(zeros)),'minor_evaluations':expected_minor}
  divisions+=d*(h-len(zeros));minors+=expected_minor
  assert a['not_complete_native_or_bit_cost'] is True
 for row,(name,inp,ex) in zip(r['rows'][12:],invalid,strict=True):
  assert row['name']==name and row['pass'] is True and row['input']==inp and row['expected']==row['exception']['type']==ex and 'result' not in row
  # Invalid categories are independently identified from the saved input; no primitive execution.
  if name=='float_coefficient':assert type(row['input'][0][0]) is float
  if name=='bool_coefficient':assert type(row['input'][0][0]) is bool
 events=[json.loads(x) for x in (BASE/'controls_progress.jsonl').read_text().splitlines()]
 expected=[{'phase':'start'}]
 for row in r['rows']:expected += [{'phase':'attempt','name':row['name']},{'phase':'result','name':row['name'],'pass':True}]
 expected += [{'phase':'complete'}];assert events==expected and len(events)==42
 out={'complete':True,'pass_all':True,'task':'T229 independent saved exact control certificates','pins':pins,'checker_sha256':sha(Path(__file__)),'valid':12,'invalid':8,'identity_forced':7,'inconclusive':5,'pair_certificates':pairs,'normalization_divisions':divisions,'minor_evaluations':minors,'progress_events':42,'review_cpu_seconds':time.process_time()-t,'author_cpu_seconds':r['cpu_seconds'],'author_or_target_calls':0,'scope':'Exact saved column, positive-ray classes, minor and ratio certificates, fixture outcomes, pin/progress and source-expression partial ledger checks.','limitations':['Readout obstruction is sufficient only for global positive-monomial unchanged-logit weight automorphisms.','Inconclusive inputs do not establish symmetry; input/recurrent constraints remain unchecked.','No argmax/common-shift/domain/approximate/noninvertible equivalence or target-model conclusion.','Partial setup counts exclude complete native arithmetic and bit costs.']}
 OUT.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'complete':True,'sha256':sha(OUT),'cpu':out['review_cpu_seconds']}))
if __name__=='__main__':main()
