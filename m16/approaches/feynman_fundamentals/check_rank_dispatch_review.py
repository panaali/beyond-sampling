"""Saved SVD ABI/parity and nested iteration-observation audit, no cost bound."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
import json,math,time
start=time.process_time();H=Path(__file__).parent;P=Path('/workspace/work/recover-small_h4_4/m16/approaches/rank_cost_review');B=json.loads((P/'baseline.json').read_text());I=json.loads((P/'instrumented.json').read_text());C=json.loads((P/'counted_instrumented.json').read_text());assert len(B['rows'])==len(I['rows'])==len(C['rows'])==114;shapes=set();calls=0
for b,i,c in zip(B['rows'],I['rows'],C['rows']):
 assert b=={k:v for k,v in i.items() if k!='spy'}=={k:v for k,v in c.items() if k not in ('spy','nested')};s=i['spy'];m=b['m'];shapes.add(m);k=min(m,4)
 assert s['calls']==s['jobN']==s['success']==6 and s['queries']==s['executions']==3 and s['m']==m and s['n']==4 and s['lda']>=m and s['last_info']==0;calls+=s['calls']
 expected={'zero':(0,0),'rank_one':(1,1),'diagonal':(k,k),'near_threshold':(min(k,2),1),'large_diagonal':(k,k),'small_diagonal':(k,0)}[b['name']];assert (b['rank_default'],b['rank_tol1e12'])==expected
 nested=c['nested'];assert not nested['q1_failure'] and not nested['q2_failure'] and not nested['q3_negative_deltas']
DB=json.loads((P/'dense_baseline.json').read_text());DI=json.loads((P/'dense_instrumented.json').read_text());q3=iters=0;maximum=0
for b,i in zip(DB['rows'],DI['rows']):
 assert b=={k:v for k,v in i.items() if k!='nested'};k=min(b['m'],4);expected=sorted((2+2*math.cos(j*math.pi/(k+1)) for j in range(1,k+1)),reverse=True);got=[float.fromhex(x) for x in b['singular_values_hex']];assert max(abs(a-v) for a,v in zip(expected,got))<2e-14
 n=i['nested'];assert not n[3] and not n[7] and not n[11];q3+=n[10];iters+=n[8];maximum=max(maximum,n[9])
assert len(shapes)==19 and calls==684 and q3==78 and iters==69 and maximum==1
out=dict(analytic_shape_fixtures=114,observed_M_by_4_shapes=sorted(shapes),gesdd_queries_and_executions=calls,jobN_and_info0=True,dense_analytic_fixtures=len(DB['rows']),q3_calls=q3,q3_observed_iteration_deltas=iters,max_observed_delta=maximum,review_cpu=time.process_time()-start,scope='Transparent installed ABI and saved parity only. Counted ITER-delta repair avoids undefined/early-return Q2 workspace values. Observed iterations do not prove any worst-case cost bound.')
(H/'rank_dispatch_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
