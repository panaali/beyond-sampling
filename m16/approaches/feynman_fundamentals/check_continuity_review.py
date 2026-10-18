"""Exact checkpoint slope certificate and independent side-error arithmetic."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import hashlib,json,math,time
import torch
start=time.process_time();p=Path('/workspace/work/recover-small_h4_4/m16/approaches/penultimate_continuity/pilot.json');d=json.loads(p.read_text());path=Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth');assert hashlib.sha256(path.read_bytes()).hexdigest()==d['checkpoint_sha256'];w=torch.load(path,map_location='cpu',weights_only=True);wi=[F(float(v)) for v in w['rnn.weight_ih_l0'][:,0]];wh=[[F(float(v)) for v in r] for r in w['rnn.weight_hh_l0']];wo=[[F(float(v)) for v in r] for r in w['linear.weight']];M=max(sum(map(abs,r)) for r in wh);G=F(0);cert=d['certificate'];assert M==F(cert['M'])
for i,v in enumerate(wi):assert v!=0;term=2*M**3/abs(v);G+=term;assert F(cert['gates'][i]['slope'])==v and F(cert['gates'][i]['coefficient'])==term
checks=0
for support in product((False,True),repeat=2):
 if not any(support):continue
 for j in(0,1,3):
  delta=[a-b for a,b in zip(wo[2],wo[j])];slope=sum(delta[i]*wi[i] for i in range(2) if support[i]);assert slope!=0;term=2*sum(map(abs,delta))*M**3/abs(slope);G+=term;row=next(r for r in cert['margins'] if tuple(r['support'])==support and r['competitor']==j);assert F(row['slope'])==slope and F(row['coefficient'])==term;checks+=1
assert G==F(cert['G']) and cert['sufficient'];Phi=lambda x:.5*math.erfc(-x/math.sqrt(2));L,U=Phi(1)-Phi(-6),Phi(-1)-Phi(-6);C=float(G)/min(L,U);assert abs(C-d['lipschitz_coefficient_float'])<1e-5
rows=[]
for r in d['rows']:
 differences=[[F(float(x))-F(float(y)) for x,y in zip(a,b)] for a,b in zip(r['old_sides'],r['exact_sides'])]
 for i in range(2):
  for j in range(2):assert abs(float(differences[i][j])-r['side_errors'][i][j])<2e-16
 parts=[differences[0][1]/2,differences[1][0]/2];assert all(abs(float(x)-y)<1e-16 for x,y in zip(parts,r['need1_error_parts']));rows.append({'prefix':r['prefix'],'need1_error_parts':list(map(float,parts)),'need1_error':float(sum(parts))})
out={'exact_gate_slopes':2,'exact_readout_slopes':checks,'M':float(M),'G':float(G),'conditional_Lipschitz_coefficient':C,'side_errors':rows,'cpu_seconds':time.process_time()-start,'scope':'sufficient exact continuity theorem; displayed coefficient not outward and too loose for useful mesh error control'}
Path(__file__).with_name('continuity_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
