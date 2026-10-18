"""Verify four saved cap diagnostics without source cap or mass calls."""
from pathlib import Path
from fractions import Fraction as F
import ast,hashlib,json,math,time
P=Path('/workspace/work/recover-small_h4_4/review/split_attribution_review');H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 start=time.process_time();d=json.load(open(P/'cap_results.json'));gate=json.load(open(P/'cap_gate.json'))
 for f,h in gate['pins'].items():assert sha(P/f)==h,f
 assert d['gate_sha256']==sha(P/'cap_gate.json') and d['source_sha256']==sha(P/'frozen/prune.py')
 node=next(n for n in ast.parse((P/'frozen/prune.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='cap_bound');assert hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()==d['ast_sha256']
 rows=d['controls'];assert [r['name'] for r in rows]==['orthant2','orthant3','narrow3','zero_axis2']
 assert rows[0]['rays']==[[1.,0.],[0.,1.]] and rows[1]['rays']==[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
 for i,target in [(0,.25),(1,(1-1/math.sqrt(3))/2)]:assert abs(rows[i]['value']-target)<=5e-15 and rows[i]['analytic_cap']==target and rows[i]['exact_cone_mass']==[.25,.125][i]
 e=F(1,2**30);R=[[F(x) for x in row] for row in rows[2]['rays']];assert R==[[1,0,0],[1,e,0],[1,0,e]]
 lower=e*e/41472;r=rows[2];assert F(r['e'])==e and F(r['determinant'])==e*e and F(r['box_volume'])==e*e/64 and F(r['strict_mass_lower'])==lower
 assert F(r['value'])==0<lower and r['below_strict_lower'] is True
 assert F(1)-F(1,4)-F(1,4)==F(1,2) and 4+2*(e/4)**2<6
 z=rows[3];assert z['rays']==[[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]] and z['zero_axis'] and z['value']==0 and z['warnings']==['invalid value encountered in divide']
 for r in rows:
  nr=len(r['rays']);dim=len(r['rays'][0]);assert r['finite'] and math.isfinite(r['value']) and r['source_ops']==[nr*(dim+1)+2*dim+30]
 progress=[json.loads(x) for x in (P/'cap_progress.jsonl').read_text().splitlines()]
 assert progress==[{'state':'started','gate':sha(P/'cap_gate.json')}]+[{'state':'case_completed','name':r['name']} for r in rows]+[{'state':'completed','result_sha256':sha(P/'cap_results.json')}]
 result={'complete':True,'source_commit':'cf0d5779','raw_sha256':sha(P/'cap_results.json'),'gate_sha256':sha(P/'cap_gate.json'),'checker_sha256':sha(Path(__file__)),'control_count':4,'source_proxy_total':sum(r['source_ops'][0] for r in rows),'narrow_stored_cone_cap':0,'narrow_exact_strict_mass_lower':str(lower),'zero_axis_returns_finite_zero_with_warning':True,'author_cpu':d['cpu_seconds'],'review_cpu':time.process_time()-start,'cap_model_geometry_mass_calls':0,'conclusion':'The saved narrow floating cap is not an upper bound on the exact positive mass of its stored cone. The cap theorem remains valid in exact arithmetic; this implementation is not generally outward.','limitations':['Counterexample mass lower bound is extremely small; no trained-panel accuracy failure follows.','Zero-axis fixture tests an undefined-direction domain, not asserted reachable in ordering chambers.','Source operation charge is inherited proxy, not hardware count.']}
 (H/'split_cap_saved_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'complete':True,'lower':str(lower),'review_cpu':result['review_cpu']}))
if __name__=='__main__':main()
