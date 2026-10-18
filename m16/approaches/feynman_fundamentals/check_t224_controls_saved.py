"""Exact saved projection/SSE controls, no author primitive or matrix library calls."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import ast,hashlib,json,time,math
P=Path('/workspace/work/recover-small_h2_56/review/lookup_nonnegative_projection');H=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def val(n):
 if isinstance(n,ast.Constant):return n.value
 if isinstance(n,(ast.List,ast.Tuple)):return [val(x) for x in n.elts]
 if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return -val(n.operand)
 if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Pow):return val(n.left)**val(n.right)
 raise ValueError(ast.dump(n))
def frac(x):return F(float(x))
def fold(a):
 s=float(a[0])
 for v in a[1:]:s=s+float(v)
 return s
def logits(a,w):return [[fold([float(x)*float(y) for x,y in zip(row,v,strict=True)]) for v in w] for row in a]
def labels(a):return [max(range(len(row)),key=lambda j:row[j]) for row in a]
def exact_pair(x):return [str(x.numerator),str(x.denominator)]
ZERO_READOUT_DIFFERENCES=0
def equalfloat(a,b,zero_equiv=False):
 global ZERO_READOUT_DIFFERENCES
 if isinstance(a,list):assert len(a)==len(b);[equalfloat(x,y,zero_equiv) for x,y in zip(a,b,strict=True)]
 else:
  if zero_equiv and a==b==0 and float(a).hex()!=float(b).hex():ZERO_READOUT_DIFFERENCES+=1
  else:assert float(a).hex()==float(b).hex(),(a,b)
def main():
 start=time.process_time();r=read(P/'controls/results.json');p=read(P/'protocol.json');g=read(P/'execution_gate.json')
 assert r['status']=='complete' and r['pass_all'] and r['gate']==g and g['clearance'] and r['protocol_sha256']==g['protocol_sha256']==sha(P/'protocol.json') and r['runtime']==p['runtime']
 for path,digest in {**p['pins'],**g['pins']}.items():assert sha(path)==digest
 tree=ast.parse((P/'controls.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fixtures');fixtures=val(next(n.value for n in fn.body if isinstance(n,ast.Return)))
 assert len(fixtures)==len(r['rows'])==10 and len(r['guards'])==6
 expected_events=[dict(phase='attempt',stage='runtime'),dict(phase='result',stage='runtime')]
 counts=Counter();nrows=0;coords=0;worse=[];better=[];drops=[]
 for fixture,row in zip(fixtures,r['rows'],strict=True):
  name,h,z,w,truth,expected=fixture
  assert row['name']==name and row['truth']==truth and row['expected_labels']==expected and row['status']=='complete' and row['passed']
  equalfloat(row['hidden'],h);equalfloat(row['lookup'],z);equalfloat(row['wo'],w)
  projected=[[float(v) if v>0 else 0.0 for v in vrow] for vrow in z];equalfloat(row['projected'],projected)
  act=logits(h,w);base=logits(z,w);new=logits(projected,w)
  equalfloat(row['actual_logits'],act,True);equalfloat(row['baseline_logits'],base,True);equalfloat(row['projected_logits'],new,True)
  ll=[labels(v) for v in (act,base,new)];assert row['labels']==ll==expected
  for i,(hr,zr,pr,record) in enumerate(zip(h,z,projected,row['exact_sse'],strict=True)):
   before=sum(((frac(a)-frac(b))**2 for a,b in zip(zr,hr)),F(0));after=sum(((frac(a)-frac(b))**2 for a,b in zip(pr,hr)),F(0))
   drop=[frac(a)*frac(a)-2*frac(a)*frac(b) if a<0 else F(0) for a,b in zip(zr,hr)]
   assert record==dict(before=exact_pair(before),after=exact_pair(after),drop=[exact_pair(v) for v in drop]) and before-after==sum(drop,F(0)) and all(v>=0 for v in drop)
   equalfloat(row['sse_before'][i],float(before));equalfloat(row['sse_after'][i],float(after));drops.append(exact_pair(before-after))
   if ll[1][i]==truth[i] and ll[2][i]!=truth[i]:worse.append([name,i])
   if ll[1][i]!=truth[i] and ll[2][i]==truth[i]:better.append([name,i])
   nrows+=1;coords+=len(hr)
  n=len(h);width=len(h[0]);C=len(w);expectedcounts=dict(comparisons=n*width,subtractions=2*n*width,multiplications=n*width*(C+2),additions=n*(width-1)*(C+2))
  assert row['named_expression_counts']==expectedcounts;counts.update(expectedcounts)
  expected_events.extend([dict(phase='attempt',name=name),dict(phase='result',name=name,passed=True)])
 expectedguards=[('nan_lookup','ValueError'),('infinite_readout','ValueError'),('negative_actual','ValueError'),('shape_mismatch','ValueError'),('empty_batch','ValueError'),('overflow_sse','FloatingPointError')]
 for row,(name,error) in zip(r['guards'],expectedguards,strict=True):
  assert row==dict(name=name,expected=error,caught=error,passed=True)
  expected_events.extend([dict(phase='attempt',name=name),dict(phase='result',**row)])
 expected_events.append(dict(phase='complete'))
 assert [json.loads(v) for v in (P/'controls/progress.jsonl').read_text().splitlines()]==expected_events and len(expected_events)==35 and nrows==11
 assert worse==[['worsens_strict',0],['worsens_tie',0]] and better==[['improves',0]]
 out=dict(complete=True,pass_all=True,raw_commit='06ff0de6',pins={n:sha(P/n) for n in ['controls/results.json','controls/progress.jsonl','protocol.json','execution_gate.json','primitive.py','controls.py']},checker_sha256=sha(__file__),readout_signed_zero_differences=ZERO_READOUT_DIFFERENCES,reviewer_correction_sha256=sha(H/'t224_reviewer_correction.json'),fixtures=10,rows=nrows,coordinates=coords,guards=6,progress_events=35,exact_sse_drop=drops,partial_counts=dict(counts),task_improvements=better,task_regressions=worse,review_cpu=time.process_time()-start,author_cpu=r['cpu_seconds'],new_author_calls=0,scope='Fixed synthetic saved-array projection, exact coordinate SSE reduction identity, independent scalar stored-float readout/first-index labels (readout zero signs treated equivalent), partial expression counts and progress/pins pass. Recorded guards checked without invalid-path replay. Hidden SSE monotonicity does not imply classification improvement: both strict and tied worsening fixtures retained. Cancellation readout uses its actual binary64 fold, not an exact-real dot claim. No trained panel, runtime/full-cost or deployable estimator conclusion.')
 (H/'t224_controls_saved_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
