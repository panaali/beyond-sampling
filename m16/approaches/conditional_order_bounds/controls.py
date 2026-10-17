"""Frozen scalar identities versus direct unnormalized Gaussian integrals."""
from pathlib import Path
import hashlib,json,time
import mpmath as mp
from scalar import absolute_mgf,absolute_mean,expected_allowance,bound_at_rate
HERE=Path(__file__).resolve().parent

def reference(b,a,s):
 # This reference uses neither Phi/erfc nor author's mean/MGF functions.
 density=lambda x:mp.exp(-x*x/2)
 cuts=[-mp.inf,a,b] if a<b else [-mp.inf,b]
 z=mp.quad(density,[-mp.inf,b])
 return mp.quad(lambda x:mp.exp(s*abs(x-a))*density(x),cuts)/z,mp.quad(lambda x:abs(x-a)*density(x),cuts)/z

def main():
 start=time.process_time();out=HERE/'controls.json';progress=HERE/'controls_progress.jsonl';assert not out.exists() and not progress.exists()
 P=json.loads((HERE/'protocol.json').read_text())
 for p,h in P['pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 result={'complete':False,'rows':[],'domain':[],'import_and_prior_cpu_seconds':start,'protocol_sha256':hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()}
 def record(x):
  with progress.open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n')
 def save():
  result['cpu_seconds']=time.process_time()-start;result['process_cpu_seconds']=time.process_time();out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 try:
  for case in P['cases']:
   record({'stage':'attempt','case':case});b,a,s=case
   with mp.workdps(80):lo=(mp.nstr(absolute_mgf(b,a,s),80),mp.nstr(absolute_mean(b,a),80))
   with mp.workdps(120):hi=(mp.nstr(absolute_mgf(b,a,s),120),mp.nstr(absolute_mean(b,a),120))
   with mp.workdps(100):rr=reference(*map(mp.mpf,case));refs=[mp.nstr(x,100) for x in rr]
   with mp.workdps(140):
    errors=[abs(mp.mpf(x)-mp.mpf(y)) for x,y in zip(hi,refs)]
    assert all(e<=mp.mpf('1e-65')*(1+abs(mp.mpf(x))) for e,x in zip(errors,hi))
    assert all(abs(mp.mpf(x)-mp.mpf(y))<=mp.mpf('1e-65')*(1+abs(mp.mpf(x))) for x,y in zip(hi,lo))
    mean=absolute_mean(b,a);assert mean>0
    # Fixed nontriviality discriminator, no lambda optimization/search.
    high=mean+mp.mpf(1)/4;low=mean-mp.mpf(1)/4
    improves=bound_at_rate(b,[a],[1],high,'0.0001');assert improves<1
    tested=[]
    for lam in P['precheck_rates']:
     t=mp.mpf(lam);raw=-t*low+mp.log(absolute_mgf(b,a,t));assert raw>=-mp.mpf('1e-100')
     tested.append(mp.nstr(raw,110))
    if s=='0':assert mp.mpf(hi[0])==1
    if b==a=='0':assert abs(mp.mpf(hi[0])-2*mp.exp(mp.mpf(s)**2/2)*(1+mp.erf(mp.mpf(s)/mp.sqrt(2)))/2)<=mp.mpf('1e-110')
    row={'case':case,'value80':lo,'value120':hi,'reference100':refs,'errors':[mp.nstr(x,110) for x in errors],
     'margin_above_mean':mp.nstr(high,110),'bound_fixed_rate':mp.nstr(improves,110),'below_mean_log_values':tested,'passed':True}
   result['rows'].append(row);record({'stage':'result','row':row});save()
  bad=[('negative_rate',lambda:absolute_mgf(0,0,-1)),('nonfinite_bound',lambda:absolute_mgf('inf',0,1)),('negative_weight',lambda:expected_allowance(0,[0],[-1]))]
  for name,fn in bad:
   try:fn()
   except ValueError:row={'name':name,'status':'rejected'}
   else:raise AssertionError(name)
   result['domain'].append(row);record({'stage':'domain','row':row})
  with mp.workdps(100):
   zz=[bound_at_rate(0,[0],[0],m,1) for m in (-1,0,1)];assert zz==[1,1,0]
   assert expected_allowance(0,[0],[0])==0
  row={'name':'zero_allowance','margins':[-1,0,1],'bounds':[str(x) for x in zz],'status':'passed'};result['domain'].append(row);record({'stage':'domain','row':row})
  assert len(result['rows'])==8 and len(result['domain'])==4
  result['complete']=True;save();print(json.dumps({k:v for k,v in result.items() if k not in ('rows','domain')}))
 except BaseException as e:
  result['failure']=repr(e);record({'stage':'failure','error':repr(e)});save();raise
if __name__=='__main__':main()
