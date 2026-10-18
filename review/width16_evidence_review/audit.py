"""Independent saved contraction and four synthetic center-sign controls; no author imports."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter,defaultdict
import hashlib,json,math,time,resource,signal
import numpy as np
H=Path(__file__).resolve().parent
A=H/'frozen/m16/approaches/width16_probe'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def pack(x):
 if isinstance(x,F):return [str(x.numerator),str(x.denominator)]
 if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [pack(v) for v in x]
 return x
def same(a,b):
 assert math.isfinite(float(a)) and math.isfinite(float(b))
 assert math.isclose(float(a),float(b),rel_tol=5e-12,abs_tol=1e-16),(a,b)
def samples(q,n):
 k=round(F(q)*n);assert 0<=k<=n and abs(F(q)-F(k,n))<=F(math.ulp(q))
 return dict(inferred_hits=k,n=n,proportion=q,not_literal_saved_hits=True)
def controls():
 fixtures=[('regular_positive',[[F(1),F(0)],[F(1),F(1)]],1,1),('regular_negative',[[F(0),F(1)],[F(1),F(1)]],-1,-1),('narrow_positive',[[F(1),F(1)],[F(1)+F(1,2**52),F(1)]],1,0),('narrow_negative',[[F(1)-F(1,2**53),F(1)],[F(1),F(1)]],-1,0)]
 rows=[]
 for name,R,sgn,want_float in fixtures:
  exact=tuple(sum((r[j] for r in R),F(0))/2 for j in range(2));m=exact[0]-exact[1]
  r=np.array(R,dtype=np.float64);assert [[F(v) for v in row] for row in r]==R
  c=r.mean(0);f=float(c[0]-c[1]);det=R[0][0]*R[1][1]-R[0][1]*R[1][0]
  assert det and (m>0)-(m<0)==sgn and (f>0)-(f<0)==want_float
  rays=[q[0]-q[1] for q in R];assert all(sgn*v>=0 for v in rays) and any(sgn*v>0 for v in rays)
  rows.append(dict(name=name,rays=R,determinant=det,exact_mean=exact,exact_margin=m,float_mean=[float(v).hex() for v in c],float_margin=f,exact_gate=m>0,float_gate=f>0))
 assert len({r['exact_gate'] for r in rows[-2:]})==2 and len({r['float_gate'] for r in rows[-2:]})==1
 return rows

def main():
 resource.setrlimit(resource.RLIMIT_CPU,(19,20));signal.signal(signal.SIGXCPU,lambda *_:(_ for _ in ()).throw(TimeoutError('19CPU')))
 dst=H/'results.json';assert not dst.exists();start=time.process_time();out=dict(complete=False,rows=[],controls=[],real_estimator_model_mass_MC_calls=0)
 def save():dst.write_text(json.dumps(pack(out),indent=2,allow_nan=False)+'\n')
 save()
 try:
  p=load(H/'protocol.json');assert p['source_gate']!='pending'
  for path,h in p['pins'].items():assert sha(path)==h,path
  manifest=load(H/'manifest.json')
  for f in manifest['files']:assert sha(H/f['snapshot_path'])==f['sha256']
  assert np.__version__==p['runtime']['numpy_version'] and sha(np._core._multiarray_umath.__file__)==p['runtime']['numpy_core_sha256']
  out['controls']=controls();save()
  summary=load(A/'summary.json');mc=load(A/'mc.json');mc2=load(A/'mc2.json');assert len(summary)==len(mc)==32 and len(mc2)==4 and set(summary)==set(mc)
  files=sorted((A/'res').glob('*.json'));assert len(files)==55
  for path in files:
   r=load(path);assert r['stopped'] is None and r['acc'] is not None and 0<=r['acc']<=1
   for k in ('ops','cells','cut','resolved','simplices'):assert type(r[k]) is int and r[k]>=0
  raw=dict(files=len(files),stopped=0,all_nonnegative_source_counters=True)
  fieldschecked=Counter();groups=defaultdict(list)
  for key,row in sorted(summary.items()):
   out['active']=key;save();h,t,name=key.split('_',2);width=int(h[1:]);T=int(t[1:]);assert (row['H'],row['T'],row['name'])==(width,T,name)
   resolved={}
   for mode in ('exact','recipe108'):
    path=A/'res'/f'{key}_{mode}.json'
    if path.exists():r=load(path);scope='local'
    else:
     assert (width,T,mode)==(8,4,'recipe108');path=H/'frozen/m16/approaches/estT_fix/rescore'/f'{name}.json';r=load(path)[mode];scope='rescore_fallback'
    resolved[mode]=dict(raw=r,path=str(path.relative_to(H)),source=scope,sha256=sha(path))
   ex=resolved['exact']['raw'];rc=resolved['recipe108']['raw'];P=F(ex['acc']);Q=F(rc['acc']);d=Q-P
   assert 0<P<1 and 0<=Q<=1
   fpe=F(6000*2*T*width*(width+2),5760);N=F(rc['ops'])/fpe;ratio=d*d*N/(P*(1-P))
   assert row['recipe_stopped']==rc.get('stopped') is None
   for k,rawk in [('exact','acc'),('exact_ops','ops'),('exact_cells','cells'),('exact_leafsum_minus_1','leafsum_minus_1'),('regions_exact','regions_exact')]:assert row[k]==ex[rawk];fieldschecked[k]+=1
   for k,rawk in [('est','acc'),('ops','ops'),('cells','cells')]:assert row[k]==rc[rawk];fieldschecked[k]+=1
   assert row['recipe_cpu']==rc.get('cpu',rc.get('sec'))
   for k in ('mc1e7','regions_lb_2e5','regions_lb_2e6'):assert row[k]==mc[key][k]
   reg=ex['regions_exact'] or mc[key]['regions_lb_2e6'];assert type(reg) is int and reg>0
   assert 0<mc[key]['regions_lb_2e5']<=mc[key]['regions_lb_2e6']
   cpr=F(rc['cells'],reg);samplecpr=F(rc['cells'],mc[key]['regions_lb_2e5']);same(row['ratio'],ratio);same(row['err_pp'],100*d);same(row['cells_per_region'],cpr);same(row['cells_per_region_lb2e5'],samplecpr)
   q=F(mc[key]['mc1e7']);z=float(P-q)/math.sqrt(float(q*(1-q)/10_000_000));same(row['z_exact_mc'],z)
   reference=[samples(mc[key]['mc1e7'],10_000_000)]
   if key in mc2:
    q2=F(mc2[key]);assert row['mc2_2e7']==mc2[key];same(row['z_exact_mc2'],float(P-q2)/math.sqrt(float(q2*(1-q2)/20_000_000)));reference.append(samples(mc2[key],20_000_000))
   else:assert 'mc2_2e7' not in row
   cs=2*T*width*(width+2)+T*width+3*T-2;conditional_t196=ratio*2*fpe/cs
   result=dict(key=key,H=width,T=T,name=name,source_paths={k:{z:v[z] for z in ('path','source','sha256')} for k,v in resolved.items()},recipe_cells=rc['cells'],exact_cells=ex['cells'],floating_leaf_patterns=reg,sampled_patterns_2M=mc[key]['regions_lb_2e6'],recipe_cells_per_pattern=cpr,recipe_ops_per_exact_ops=F(rc['ops'],ex['ops']),published_ratio=ratio,conditional_T196_2ops_ratio=conditional_t196,canonical_sampler=cs,exact_accuracy=P,recipe_accuracy=Q,error=d,exact_leafsum_defect=ex['leafsum_minus_1'],reference_metadata=reference,z_exact_mc=z)
   out['rows'].append(result);groups[(width,T)].append(result);save()
  out['groups']=[dict(H=w,T=t,models=len(rs),clean=sum('noisy' not in r['name'] for r in rs),noisy_seed0=sum('noisy' in r['name'] for r in rs),min_cells_per_pattern=min(r['recipe_cells_per_pattern'] for r in rs),max_cells_per_pattern=max(r['recipe_cells_per_pattern'] for r in rs),max_original_ratio=max(r['published_ratio'] for r in rs),max_conditional_T196_ratio=max(r['conditional_T196_2ops_ratio'] for r in rs),min_recipe_ops_per_exact_ops=min(r['recipe_ops_per_exact_ops'] for r in rs),max_recipe_ops_per_exact_ops=max(r['recipe_ops_per_exact_ops'] for r in rs)) for (w,t),rs in sorted(groups.items())]
  out.update(complete=True,active=None,raw=raw,fields_checked=dict(fieldschecked),protocol_sha256=sha(H/'protocol.json'),source_sha256=sha(__file__),scope=p['scope'])
 except BaseException as e:out['failure']=repr(e);raise
 finally:out['cpu_seconds']=time.process_time()-start;save()
 print(json.dumps(dict(complete=out['complete'],rows=len(out['rows']),controls=len(out['controls']),cpu=out['cpu_seconds'])))
if __name__=='__main__':main()
