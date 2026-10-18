"""Independent exact contraction of saved split-child constant bounds."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from itertools import permutations
import hashlib,json,math,time
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/gauss_legendre_enclosure/perturbation/split')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f(v):return F(int(v[0]),int(v[1]))
def det(R):
 out=F()
 for perm in permutations(range(5)):
  t=F((-1)**sum(perm[i]>perm[j] for i in range(5) for j in range(i+1,5)))
  for i,j in enumerate(perm):t*=R[i][j]
  out+=t
 return out
start=time.process_time();p=read(P/'protocol.json');r=read(P/'results.json')
for path,h in p['pins'].items():assert sha(path)==h
old=read(p['constant_source']);children=read(p['children_source'])
assert r['complete'] and len(r['rows'])==66 and len(r['groups'])==33 and r['tensor_mass_model_geometry_MC_calls']==0
constants={c['order']:(f(c['epsilon']),f(c['source_weight_sum']),f(c['delta_L1'])) for c in old['constants']}
rad={}
for c,s in zip(children['children'],r['exact_setup']):
 key=(c['fixture'],c['side']);assert key not in rad and key==(s['fixture'],s['side'])
 R=[[F(float.fromhex(x)) for x in row] for row in c['rays_hex']]
 G=[[sum((R[i][k]*R[j][k] for k in range(5)),F()) for j in range(5)] for i in range(5)]
 assert G==[[F(x) for x in row] for row in c['certificate']['gram']] and all(x>=0 for row in G for x in row)
 a=min(G[i][i] for i in range(5))/5;M=max(G[i][i] for i in range(5));d=det(R)
 assert a>0 and d!=0 and (a,M,d)==(f(s['a']),f(s['M2']),f(s['det']))==(F(c['certificate']['a']),F(c['certificate']['b']),F(c['certificate']['det']))
 rad[key]=(a,M,d)
 assert s['counts']['rational_mul']==160 and s['counts']['rational_div']==11 and s['counts']['rational_sub']==30 and s['counts']['add']==125
values={};counts=None
for row in r['rows']:
 key=(row['fixture'],row['side']);a,M,d=rad[key];eps,s,delta=constants[row['order']];g=row['result'];assert g['status']=='bounded'
 k=math.isqrt((a.numerator<<320)//a.denominator);lo=F(k,1<<160);hi=F(k+1,1<<160)
 assert lo>0 and lo*lo<=a<hi*hi and [f(x) for x in g['sqrt_interval']]==[lo,hi]
 q5=1/(a*a*lo);q7=q5/a;L=6*q5+40*M*q7;node=s**4*eps*L;weight=delta*(s**3+s*s+s+1)*q5;total=node+weight;prob=abs(d)*total/24
 for name,v in [('q5_upper',q5),('q7_upper',q7),('derivative_sum_upper',L),('node_upper',node),('weight_upper',weight),('integral_upper',total),('prefactor_upper',abs(d)/24),('probability_upper',prob)]:assert f(g[name])==v
 if counts is None:counts=g['counts']
 assert g['counts']==counts
 q=(key[0],key[1],row['order']);assert q not in values;values[q]=prob
assert set(values)=={(label,side,n) for label in p['fixtures'] for side in (0,1) for n in (6,10,16)}
maxbound=F();rescued=0
for g in r['groups']:
 label,n=g['fixture'],g['order'];v=values[label,0,n]+values[label,1,n]
 prior=next(x for x in old['rows'] if (x['fixture'],x['order'])==(label,n))['result']['probability_upper']
 assert g['status']=='bounded' and v==f(g['upper']) and g['original_upper']==prior and g['below_existing_1e_10']==(v<=F(1,10**10))
 assert v<=F(1,10**10);rescued+=f(prior)>F(1,10**10);maxbound=max(maxbound,v)
progress=[json.loads(x) for x in (P/'progress.jsonl').read_text().splitlines()]
assert len(progress)==22+66+66+33 and sum(x.get('stage')=='child_attempt' for x in progress)==22 and sum(x.get('stage')=='bound_attempt' for x in progress)==66
assert r['exact_parent_sum_additions_allowance']==66 and r['source_constants_new_mapping_ops']==0 and rescued==12
out=dict(complete=True,children=22,child_bounds=66,parent_sums=33,all_below_diagnostic=True,previously_loose_rescued=rescued,max_upper=float(maxbound),max_upper_exact=str(maxbound),bound_category_counts_each=counts,results_sha256=sha(P/'results.json'),protocol_sha256=sha(P/'protocol.json'),review_cpu=time.process_time()-start,new_mass_model_or_tensor_calls=0,scope='Exact saved Gram/determinants, acute radial premises and all sensitivity expressions independently reconstructed; inherited node constants already reviewed. Ideal-real constant drift only, excluding ideal tail and floating map/kernel/determinant/assembly. Peak bit counts describe observed ledger operations only.')
Path(__file__).with_name('split_constant_bounds_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
