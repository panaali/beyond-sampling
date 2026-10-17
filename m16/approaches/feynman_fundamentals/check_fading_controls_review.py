"""Saved exact region-map certificates and analytic invariances, no author calls."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,hashlib,resource,time
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/fading_influence');O=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();dot=lambda a,b:sum(x*y for x,y in zip(a,b,strict=True))
def unpack(x):
 if isinstance(x,list):
  if len(x)==2 and all(isinstance(v,str) and v.lstrip('-').isdigit() for v in x):return F(*map(int,x))
  return [unpack(v) for v in x]
 if isinstance(x,dict):return {k:unpack(v) for k,v in x.items()}
 return x
outpath=O/'fading_controls_review_results.json';assert not outpath.exists();g=read(P/'control_protocol.json');d=unpack(read(P/'controls.json'));assert d['complete'] and len(d['rows'])==12
for p,h in g['pins'].items():assert sha(P/p)==h
progress=read(P/'controls.json');events=[json.loads(x) for x in (P/'controls_progress.jsonl').read_text().splitlines()];assert len(events)==104
counts=Counter();radiuschecks=0;maps={};envelopes={}
for row in d['rows']:
 m=row['model'];w=row['witness'];H,T,D=m['H'],m['T'],m['D'];assert T==4;x=w['x'];wi=[F(z,D) for z in m['wi']];wh=[[F(z,D) for z in r] for r in m['wh']];wo=[[F(z,D) for z in r] for r in m['wo']]
 # Region map from the SAVED mask, checking all defining inequalities at x.
 J=[[F(0)]*T for _ in range(H)];radii=[];zeros=[];boundaries=[]
 for t in range(T):
  B=[[sum(wh[i][k]*J[k][j] for k in range(H))+(wi[i] if j==t else 0) for j in range(T)] for i in range(H)]
  for i,b in enumerate(B):
   v=dot(b,x);assert (v>0)==w['history'][t][i]
   if not any(b):zeros.append([t,i])
   elif v==0:boundaries.append([t,i])
   else:radii.append(abs(v)/(2*sum(map(abs,b))))
  J=[b if w['history'][t][i] else [F(0)]*T for i,b in enumerate(B)]
 L=[[sum(wo[a][i]*J[i][j] for i in range(H)) for j in range(T)] for a in range(T)]
 assert L==[[F(z,w['logit_denominator']) for z in r] for r in w['logit_map_numerators']]
 assert zeros==w['identically_zero_gates'] and boundaries==w['boundary_gates'];gaps=[abs(x[i]-x[j]) for i in range(T) for j in range(i)];strict=not boundaries and all(gaps);assert strict==w['strict']==w['local_derivative_certified']
 if all(gaps):radii.append(F(min(gaps),4))
 assert w['radius']==(min(radii) if strict else None)
 sensitivities=[max(L[a][j] for a in range(T))-min(L[a][j] for a in range(T)) for j in range(T)];assert sensitivities==w['sensitivities_by_input'] and sensitivities[::-1]==w['age_sensitivities']
 u=list(map(abs,wi));upper=[]
 for age,r in enumerate(row['envelope']['rows']):
  expected=[{'a':a,'b':b,'bound':dot([abs(v-z) for v,z in zip(wo[a],wo[b])],u)} for a in range(T) for b in range(a)];assert expected==r['pairs'] and r['maximum']==max(v['bound'] for v in expected);upper.append(r['maximum']);u=[dot(list(map(abs,z)),u) for z in wh]
 assert all(s<=upper[T-1-j] for j,s in enumerate(sensitivities));maps[row['name']]=sensitivities;envelopes[row['name']]=upper
 npairs=T*(T-1)//2;nz=H*T-len(zeros)-len(boundaries)
 ec=dict(mul=1+T*npairs*H+(T-1)*(H*H+1),add=T*npairs*H+(T-1)*H*H,sub=T*npairs*H)
 wc=dict(mul=T*(H*T*H+H+H*(H+2)+H*T+1)+nz+T*T*H+T*H+T*T+1,add=T*(H*T*H+H+H*(H+1)+H*T)+nz*T+T*T*H+T*H+T*T,sub=T+npairs)
 assert ec==row['envelope']['counts'] and wc==w['counts'];counts.update(ec);counts.update(wc)
 if strict:
  rc={k:T*(H*H+H)+T*H for k in ('mul','add','division')};assert row['radius_baseline_counts']==rc
  assert [(r['j'],r['sign']) for r in row['radius_checks']]==[(j,s) for j in range(T) for s in (-1,1)]
  for r in row['radius_checks']:assert r['passed'] and r['delta']==r['sign']*w['radius']/2 and r['counts']==rc;radiuschecks+=1
for rate in (F(0),F(1,2),F(1),F(2)):
 name='positive_r_'+str(rate);assert maps[name]==[3*rate**(3-j) for j in range(4)] and envelopes[name]==[3*rate**j for j in range(4)]
for name in ('zero_input','negative_reset'):assert maps[name]==[0]*4
for name in ('positive_neuron_scale','neuron_permutation','common_readout'):assert maps[name]==maps['mixed_base'] and envelopes[name]==envelopes['mixed_base']
assert radiuschecks==80
out={'pass':True,'controls':12,'strict':10,'unresolved_boundary_controls':2,'saved_radius_checks':80,'named_integer_counts':dict(counts),'result_sha256':sha(P/'controls.json'),'protocol_sha256':sha(P/'control_protocol.json'),'review_cpu':time.process_time()-start,'scope':'Exact saved-mask affine certificates, defining inequalities/radius and analytic envelope/invariances independently reconstructed. No author calls or new perturbed-forward/model execution. Finite local dependence and global upper envelope do not establish fading/merging or complete cost.'}
outpath.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
