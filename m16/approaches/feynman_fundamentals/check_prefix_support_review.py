"""Independent prefix Lipschitz/slack proof from exact saved values."""
from pathlib import Path
from fractions import Fraction as F
import json,time
start=time.process_time();base=Path('/workspace/work/recover-small_h4_4/m16/approaches/output_forgetting_complement');d=json.loads((base/'prefix_support.json').read_text());saved=json.loads((base/'pilot.json').read_text());wi=list(map(F,saved['model'][0]));wh=[list(map(F,r)) for r in saved['model'][1]];wo=[list(map(F,r)) for r in saved['model'][2]]
M=max(sum(map(abs,r)) for r in wh);L=max(map(abs,wi))*sum(M**k for k in range(6));assert M==F(d['M']) and L==F(d['L6']);eps,rho=F(d['epsilon']),F(d['rho']);assert 0<eps<=F(1,16) and 0<rho<F(1,4);count=0;exponents=[]
for index,(box,hist) in enumerate(zip(d['boxes'],saved['histories'])):
 prefix=list(map(F,box['prefix']));h=[F(0),F(0)]
 for x in prefix:h=[max(F(0),wi[i]*x+sum(wh[i][j]*h[j] for j in range(2))) for i in range(2)]
 assert h==list(map(F,hist['initial_hidden']));assert box['target']==0 and box['correct']==(hist['label']==0)
 gate_ids=set();retained=set()
 for c in hist['constraints']:
  slack=F(c['slack']);suffix_loss=rho*sum(abs(F(v)) for v in c['normal'][:-1])
  if c['kind']=='gate':coefficient=M**(c['step']+1)*L;gate_ids.add((c['step'],c['neuron']))
  else:coefficient=sum(abs(a-b) for a,b in zip(wo[hist['label']],wo[c['competitor']]))*M**4*L;retained.add(c['competitor'])
  assert slack-suffix_loss-coefficient*eps>=slack/4>0;count+=1
 assert gate_ids=={(t,i) for t in range(4) for i in range(2)}
 for k in range(10):
  if k==hist['label'] or k in retained:continue
  assert k>hist['label'] and all(not active or wo[hist['label']][j]==wo[k][j] for j,active in enumerate(hist['gates'][-1]))
 center=prefix+list(map(F,saved['word']));radii=[eps]*6+[rho]*4
 assert center[8]-radii[8]>center[0]+radii[0]
 assert all(center[0]-radii[0]>center[k]+radii[k] for k in range(10) if k not in(0,8))
 exponent=sum((v.numerator+v.denominator-1)//v.denominator+1 for v in [(abs(c)+r)**2/2 for c,r in zip(center,radii)])
 assert exponent==box['exponent'];exponents.append(exponent)
out={'exact_center_reachability':2,'whole_box_margin_checks':count,'unique_second_index':0,'factored_density_bound_exponents':exponents,'cpu_seconds':time.process_time()-start,'scope':'actual two positive-mass prefix/suffix boxes; no practical error-mass lower bound claimed'}
Path(__file__).with_name('prefix_support_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
