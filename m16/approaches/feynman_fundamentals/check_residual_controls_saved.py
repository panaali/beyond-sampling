"""Independent saved residual controls: exact rational terms, labels and scalar IEEE order."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import ast,json,hashlib,time
H=Path(__file__).resolve().parent;P=Path('/workspace/work/recover-small_h2_56/review/mechanism_residual_margin')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 start=time.process_time();p=read(P/'controls_v2_protocol.json');g=read(P/'controls_v2_gate.json');d=read(P/'controls_v2_results.json');old=read(P/'controls_results.json')
 for f,h in p['pins'].items():assert sha(f)==h,f
 assert g['verified_pins']==p['pins'] and g['protocol_sha256']==sha(P/'controls_v2_protocol.json')
 assert d['complete'] and d['source_calls']==14 and d['protocol_sha256']==sha(P/'controls_v2_protocol.json') and d['gate_sha256']==sha(P/'controls_v2_gate.json')
 assert not old['complete'] and old['source_calls']==0 and old['rows']==old['guards']==[] and "np.__version__" in old['failure']
 assert d['runtime']==dict(executable=p['invoked_python'],resolved_executable=p['python'],prefix=p['venv_prefix'],numpy_version=p['numpy_version'],numpy_file=p['numpy_file'])
 oldtree=ast.parse((P/'controls.py').read_text());newtree=ast.parse((P/'controls_v2.py').read_text());casefn=lambda t:next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='cases')
 assert ast.dump(casefn(oldtree))==ast.dump(casefn(newtree))
 specs=ast.literal_eval(next(n.value for n in casefn(newtree).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='specs' for t in n.targets)))
 names=[s[0] for s in specs]+['ordered_cancellation'];assert [r['fixture']['name'] for r in d['rows']]==names
 events=[json.loads(x) for x in (P/'controls_v2_progress.jsonl').read_text().splitlines()]
 assert Counter(e['stage'] for e in events)==dict(started=1,runtime_bound=1,case_attempt=10,case_result=10,guard_attempt=4,guard_result=4,completed=1)
 assert [e['name'] for e in events if e['stage']=='case_attempt']==[e['name'] for e in events if e['stage']=='case_result']==names
 assert events[-1]['results_sha256']==sha(P/'controls_v2_results.json') and events[1]['runtime']==d['runtime']
 prior=[json.loads(x) for x in (P/'controls_progress.jsonl').read_text().splitlines()];assert [e['stage'] for e in prior]==['started','failure']
 totals=Counter();transitions=Counter();checks=[]
 for i,row in enumerate(d['rows']):
  c=row['fixture'];out=row['output'];y=c['truth'];a=c['logits'];b=c['lookup_logits'];w=c['wo'];n=len(c['h']);assert n==len(c['hh'])==16 and len(w)==3 and all(len(v)==16 for v in w)
  if 1<=i<=8:
   name,h,hh,truth,tr=specs[i];assert c['h']==h+[0]*13 and c['hh']==hh+[0]*13 and a==h and b==hh and y==truth and c['transition']==tr
   assert w==[[int(j==k) for k in range(16)] for j in range(3)]
  if i==0:assert c==dict(name='both_correct',h=[1,3,4]+[0]*13,hh=[1,4,5]+[0]*13,logits=[-1,-3,-4],lookup_logits=[-1,-4,-5],wo=[[-int(j==k) for k in range(16)] for j in range(3)],truth=0,transition=0)
  if i==9:assert c==dict(name='ordered_cancellation',h=[0]*16,hh=[2**53,1,-2**53]+[0]*13,logits=[0,0,-1],lookup_logits=[1,0,-1],wo=[[1,1,1]+[0]*13,[0]*16,[-1,-1,-1]+[0]*13],truth=0,transition=0)
  pred=min(range(3),key=lambda j:(-a[j],j));look=min(range(3),key=lambda j:(-b[j],j));competitor=min((j for j in range(3) if j!=y),key=lambda j:(-a[j],j))
  tr=0 if pred==look==y else 1 if pred!=y and look==y else 2 if pred==y and look!=y else 3 if pred==look else 4
  assert out['network']==[pred] and out['lookup']==[look] and out['competitor']==[competitor] and out['transition']==[tr] and tr==c['transition'];transitions[tr]+=1
  terms=[(F(v)-F(u))*(F(w[y][j])-F(w[competitor][j])) for j,(u,v) in enumerate(zip(c['h'],c['hh'],strict=True))]
  assert list(map(str,terms))==row['exact_terms'] and str(sum(terms))==row['exact_sum'] and [F(v) for v in out['terms'][0]]==terms
  floats=[float(v) for v in terms];s=floats[0];ab=abs(s)
  for v in floats[1:]:s=s+v;ab=ab+abs(v)
  om=float(a[y])-float(a[competitor]);lm=float(b[y])-float(b[competitor]);delta=lm-om
  expected={'term_sum':s,'term_abs_sum':ab,'original_margin':om,'lookup_margin':lm,'margin_delta':delta,'closure_residual':delta-s}
  for k,v in expected.items():assert out[k]==[v]
  assert out['network_max_ties']==[a.count(a[pred])] and out['lookup_max_ties']==[b.count(b[look])]
  counts=dict(subtractions=36,multiplications=16,additions=30,absolute_values=16);assert out['named_arithmetic']==counts;totals.update(counts)
  if i==9:assert sum(terms)==1 and s==0 and delta-s==1
  else:assert delta-s==0
  checks.append(dict(name=c['name'],network=pred,lookup=look,competitor=competitor,transition=tr,exact_sum=str(sum(terms)),sequential_sum=s,closure=delta-s))
 guards=[dict(name=n,rejected=True,error=e) for n,e in [('hidden_shape','nonempty matching hidden matrices required'),('truth_float','integer truth indices required'),('truth_range','integer truth indices required'),('nonfinite','finite arrays required')]]
 assert d['guards']==guards
 assert [e['name'] for e in events if e['stage']=='guard_attempt']==[e['name'] for e in events if e['stage']=='guard_result']==[v['name'] for v in guards]
 out=dict(complete=True,pass_all=True,author_result_commit='94ca93af',pins={n:sha(P/n) for n in ['controls_v2_results.json','controls_v2_progress.jsonl','controls_v2_protocol.json','controls_v2_gate.json','controls_results.json','controls_progress.jsonl']},checker_sha256=sha(__file__),cases=10,guards=4,new_author_calls=0,rows=checks,named_arithmetic=dict(totals),transition_counts=dict(transitions),prior_zero_call_failure_cpu=old['cpu_seconds'],successful_control_cpu=d['cpu_seconds'],total_author_recorded_cpu=old['cpu_seconds']+d['cpu_seconds'],review_cpu=time.process_time()-start,scope='Independent saved analytic labels/competitor/exact terms/sequential floating sums/ties/counts/progress/runtime pins. Guard outcomes verified without replay. Cancellation supplied logits are bookkeeping-only and deliberately not a network-consistent fixture. No trained contraction, causal attribution, estimator or total-cost claim.')
 (H/'residual_controls_saved_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(complete=True,review_cpu=out['review_cpu'],transitions=out['transition_counts'])))
if __name__=='__main__':main()
