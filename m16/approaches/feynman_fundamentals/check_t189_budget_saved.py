from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time
P=Path('/workspace/work/recover-small_h2_78/m16/approaches/conditional_order_bounds/box_mass');O=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
start=time.process_time();r=load(P/'budget_results.json');p=load(P/'budget_protocol.json');old=load(P/'recovery_results.json')
assert r['complete'] and r['protocol_sha256']==sha(P/'budget_protocol.json')
for path,h in p['pins'].items():assert sha(Path(path))==h,path
assert len(r['controls'])==2 and len(r['models'])==2
for row in r['controls']+r['models']:
 w=F(row['width_lower']);n=row['minimal_samples'];c=row['sample_cost_T196'];B=row['minimal_integer_budget']
 assert F(row['strict_se_check_squared'])==4*n*w*w>1
 assert F(row['previous_sample_check_squared'])==4*(n-1)*w*w<=1
 assert B==n*c and B//c==n and (B-1)//c==n-1
for row,previous,H,c,B in zip(r['models'],old['models'],[4,8],[218,682],[7630,94116],strict=True):
 assert row['width_lower']==previous['integrated_accuracy_interval_width_lower'] and row['source_mass']==previous['certified_union_mass']
 assert row['sample_cost_T196']==c==4*(2*H*H+2*H)+8*H+4*H+10 and row['minimal_integer_budget']==B
assert r['named_bridge_counts']=={'rational_multiply':8,'rational_divide':4,'integer_floor_divide':4,'integer_add':4,'integer_multiply':4}
events=[json.loads(l) for l in (P/'budget_progress.jsonl').read_text().splitlines()];assert len(events)==8 and not any(e['stage']=='failure' for e in events)
out={'complete':True,'owner_commit':'fac9c843','result_sha256':sha(P/'budget_results.json'),'protocol_sha256':sha(P/'budget_protocol.json'),'minimal_integer_budgets':{'H4T4':7630,'H8T4':94116},'exact_control_and_model_thresholds':4,'review_cpu':time.process_time()-start,'scope':'Canonical protocol sampling units and certificate-width versus maximal sampling SD; not actual-error/MSE lower bound or candidate execution cost.'}
(O/'t189_budget_saved_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
