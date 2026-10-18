"""Independent saved rational identities and endpoint certificates; no author calls."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,time
B=Path('/workspace/work/recover-small_h2_56/review/two_hinge_exact_controls');O=Path(__file__).with_name('t233_exact_saved_review.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def q(v):
 assert isinstance(v,list) and len(v)==2 and all(type(x) is str for x in v)
 n,d=map(int,v);assert d>0 and str(n)==v[0] and str(d)==v[1];f=F(n,d);assert (f.numerator,f.denominator)==(n,d);return f
def main():
 assert not O.exists();start=time.process_time();p=read(B/'protocol.json');g=read(B/'gate.json');fixtures=read(B/'fixtures.json');r=read(B/'output/results.json')
 for pins in (p['pins'],g['pins']):
  for path,h in pins.items():assert sha(path)==h
 rv=read(g['review_path']);assert rv['complete'] is True and rv['pass_all'] is True and sha(g['review_path'])==g['review_sha256']
 for path,h in rv['pins'].items():assert sha(path)==h
 assert g['execution_allowed'] is True and g['allowed_attempts']==1 and g['execution_scope']=='3d372553'
 assert r['complete'] is True and r['pass_all'] is True and r['protocol_sha256']==g['protocol_sha256']==sha(B/'protocol.json') and r['gate_sha256']==sha(B/'gate.json')
 assert r['runtime']['executable']==p['python'];assert r['runtime']['limits']=={'cpu':[20,20],'address_space':[2147483648,2147483648]}
 assert len(fixtures)==len(r['rows'])==12
 counts={'points':0,'strict_interior':0,'anchor_endpoint':0,'other_endpoint':0,'outside':0,'nonzero_residual':0}
 area_records=[]
 for f,row in zip(fixtures,r['rows'],strict=True):
  assert row['name']==f['name'] and row['passed'] is True
  alpha,beta,c,d,di,dj,margin=map(F,f['parameters']);assert list(map(q,row['parameters']))==[alpha,beta,c,d,di,dj,margin]
  assert alpha*beta<0 and di>0 and dj>0 and margin>=0
  # Solve endpoints from linear equations; integrate the linear height by trapezoid area.
  u=-c/alpha;v=-d/beta;lo=min(u,v);hi=max(u,v);height_lo=abs(beta*lo+d);height_hi=abs(beta*hi+d)
  area=(height_lo+height_hi)*(hi-lo)/2
  def interval_cert(z,scale=F(1)):
   expected={'u':u,'v':v,'lower':lo,'upper':hi,'delta':hi-lo,'area':scale*area,'anchor_endpoint_residual':scale*((beta*u+d+abs(beta*u+d))/2)}
   assert set(z)==set(expected) and {k:q(val) for k,val in z.items()}==expected
  gauge=row['gauge'];interval_cert(gauge['original']);interval_cert(gauge['transformed'],dj)
  assert q(gauge['weighted_area'])==q(gauge['transformed_weighted_area'])==margin*area
  assert q(row['independent_area'])==area
  assert len(row['points'])==len(f['x'])==5
  for xstr,rec in zip(f['x'],row['points'],strict=True):
   x=F(xstr);a=alpha*x+c;b=beta*x+d;assert q(rec['x'])==x and q(rec['a'])==a and q(rec['b'])==b
   # ReLU(t)=(t+|t|)/2 independently checks both saved point branches.
   surplus=b if a<0 else F(0);relu=(b+abs(b))/2;residual=relu-surplus
   assert residual>=0 and q(rec['original']['surrogate'])==surplus and q(rec['original']['residual'])==residual
   assert q(rec['scaled']['surrogate'])==dj*surplus and q(rec['scaled']['residual'])==dj*residual
   assert (margin/dj)*q(rec['scaled']['residual'])==margin*residual
   interval_cert(rec['interval']);assert q(rec['interval_residual'])==residual
   if lo<x<hi:
    counts['strict_interior']+=1;assert residual==abs(beta*x+d)
   elif x==u:
    counts['anchor_endpoint']+=1;assert residual==relu
   elif x==v:
    counts['other_endpoint']+=1;assert residual==0
   else:
    counts['outside']+=1;assert residual==0
   counts['points']+=1;counts['nonzero_residual']+=residual!=0
  area_records.append({'name':row['name'],'area':str(area),'weighted_area':str(margin*area)})
 guards=[('bool','TypeError'),('float','TypeError'),('nan','TypeError'),('zero_alpha','ValueError'),('zero_beta','ValueError'),('same_sign','ValueError'),('zero_gauge','ValueError'),('negative_margin','ValueError')]
 assert len(r['guards'])==8
 for row,(name,ex) in zip(r['guards'],guards,strict=True):assert row=={'name':name,'passed':True,'exception':ex}
 assert r['calls']=={'point':120,'interval_point':60,'scale':12,'guard_attempts':8}
 events=[json.loads(x) for x in (B/'output/progress.jsonl').read_text().splitlines()];expected=[{'event':'attempt'}]
 for f in fixtures:expected += [{'event':'case_attempt','name':f['name']},{'event':'case_result','name':f['name'],'passed':True}]
 for name,ex in guards:expected += [{'event':'guard_attempt','name':name},{'event':'guard_result','name':name,'passed':True,'exception':ex}]
 expected += [{'event':'complete'}];assert events==expected and len(events)==42
 result={'complete':True,'pass_all':True,'task':'T233 independent saved rational residual controls','execution_scope':'3d372553','pins':{str(B/n):sha(B/n) for n in ('primitive.py','controls.py','fixtures.json','PROTOCOL.md','protocol.json','gate.json','output/results.json','output/progress.jsonl')},'dependency_pins':p['pins'],'checker_sha256':sha(Path(__file__)),'cases':12,'guards':8,'point_classification':counts,'area_checks':area_records,'author_top_level_calls':r['calls'],'progress_events':42,'resource_limits':r['runtime']['limits'],'resource_protocol_limits_recorded':True,'review_cpu_seconds':time.process_time()-start,'author_main_cpu_seconds':r['cpu_seconds'],'new_author_model_rng_gaussian_calls':0,'limitations':['Exact finite rational identities/intervals/area and positive-coordinate scaling only; no Gaussian density or conditional integral evaluated.','Density maximum constant remains symbolic; no numerical outward probability certificate claimed.','Top-level author call ledger counts only named wrapper calls, not complete recursive arithmetic or bit costs.','No target residual distribution, propagated error, accuracy, runtime advantage or cost claim.']}
 O.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'complete':True,'sha256':sha(O),'cpu':result['review_cpu_seconds']}))
if __name__=='__main__':main()
