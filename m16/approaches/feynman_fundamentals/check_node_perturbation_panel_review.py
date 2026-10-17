"""Exact saved 33-node-constant perturbation contractions, no tensor rules."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from itertools import permutations
import hashlib,json,math,time
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/gauss_legendre_enclosure/perturbation')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def f(q):return F(int(q[0]),int(q[1]))
def det(a):
 n=len(a);v=F()
 for p in permutations(range(n)):
  t=F((-1)**sum(p[i]>p[j] for i in range(n) for j in range(i+1,n)))
  for i,j in enumerate(p):t*=a[i][j]
  v+=t
 return v
start=time.process_time();r=read(P/'results.json');p=read(P/'run_protocol.json')
assert r['complete'] and r['cases']==len(r['rows'])==33 and r['tensor_mass_model_geometry_MC_calls']==0
for name,h in p['pins'].items():assert sha(Path(name))==h
nodes=read(p['nodes']);cert=read(p['certificates']);orig=read(p['original'])
assert nodes['all_bounded'] and cert['complete'] and orig['complete']
constants={}
for src,saved in zip(nodes['orders'],r['constants']):
 n=src['n'];eps=delta=s=F();assert saved['order']==n
 assert len(src['source_comparison'])==len(src['source_nodes_hex'])==len(src['source_weights_hex'])==n
 for c,xh,wh,it in zip(src['source_comparison'],src['source_nodes_hex'],src['source_weights_hex'],saved['items']):
  x=F((float.fromhex(xh)+1)/2);w=F(float.fromhex(wh)/2)
  lo,hi=[(f(v)+1)/2 for v in c['ideal_node']];wl,wu=[f(v)/2 for v in c['ideal_weight']]
  assert 0<=x<=1 and 0<=lo<=hi<=1 and w>=0 and 0<wl<=wu
  dx=max(abs(x-lo),abs(x-hi));dw=max(abs(w-wl),abs(w-wu))
  assert f(it['source_node'])==x and f(it['source_weight'])==w and f(it['node_error_upper'])==dx and f(it['weight_error_upper'])==dw
  eps=max(eps,dx);delta+=dw;s+=w
 assert f(saved['epsilon'])==eps and f(saved['delta_L1'])==delta and f(saved['source_weight_sum'])==s and abs(s-1)<=delta
 constants[n]=(eps,s,delta)
assert list(constants)==[6,10,16]
radial={}
for c,o,setup in zip(cert['certificates'],orig['fixtures'],r['exact_ray_setup']):
 assert c['fixture']==o['name']==setup['fixture']
 A=[[F(float.fromhex(x)) for x in row] for row in o['rays_hex']]
 G=[[sum((A[i][k]*A[j][k] for k in range(5)),F()) for j in range(5)] for i in range(5)]
 assert G==[[F(x) for x in row] for row in c['gram']]
 a=F(c['a']);M=max(G[i][i] for i in range(5));d=det(A)
 assert d==F(o['exact_stored_determinant']) and abs(d)==abs(F(c['det']))
 assert (f(setup['a']),f(setup['M2']),abs(f(setup['det'])))==(a,M,abs(d)) and 0<a<=M
 radial[c['fixture']]=(a,M,F(c['det']))
assert [(q['fixture'],q['order']) for q in r['rows']]==[(label,n) for label in p['fixture_ids'] for n in (6,10,16)]
small=0;max_bound=F();by=[]
for q in r['rows']:
 a,M,d=radial[q['fixture']];eps,s,delta=constants[q['order']];g=q['result']
 assert (f(q['a']),f(q['M2']),f(q['det']))==(a,M,d) and g['status']=='bounded'
 k=math.isqrt((a.numerator<<320)//a.denominator);lo=F(k,1<<160);assert lo>0
 q5=1/(a*a*lo);q7=q5/a;L=6*q5+40*M*q7
 node=s**4*eps*L;weight=delta*(s**3+s**2+s+1)*q5;total=node+weight;prob=abs(d)*total/24
 for name,v in [('q5_upper',q5),('q7_upper',q7),('derivative_sum_upper',L),('node_upper',node),('weight_upper',weight),('integral_upper',total),('probability_upper',prob),('prefactor_upper',abs(d)/24)]:assert f(g[name])==v
 passes=prob<=F(1,10**10);assert q['below_existing_1e_10_diagnostic']==passes;small+=passes;max_bound=max(max_bound,prob)
 by.append(dict(fixture=q['fixture'],order=q['order'],upper=str(prob),below_diagnostic=passes))
assert small==21 and r['source_mapping_arithmetic']==3*(6+10+16)
out=dict(complete=True,cases=33,bounded=33,below_diagnostic=small,loose=12,results_sha256=sha(P/'results.json'),
 protocol_sha256=sha(P/'run_protocol.json'),rows=by,review_cpu=time.process_time()-start,new_tensor_mass_or_model_calls=0,
 scope='All mapped constants, exact Gram/determinants, inherited radial bindings and rational sensitivity expressions independently contracted. Ideal-real source-constant drift only; loose bounds do not demonstrate actual error, and map/kernel/determinant/accumulation floating work remains excluded.')
Path(__file__).with_name('node_perturbation_panel_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
