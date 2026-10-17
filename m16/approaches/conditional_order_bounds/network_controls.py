"""One frozen seven-case run; no quadrature, checkpoints, random draws or training."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,time
import mpmath as mp
from network import model,partition,candidate,rational_strong_certificate,cdf
HERE=Path(__file__).resolve().parent
def main():
 start=time.process_time();out=HERE/'network_results.json';progress=HERE/'network_progress.jsonl'
 assert not out.exists() and not progress.exists()
 P=json.loads((HERE/'network_protocol.json').read_text())
 for p,h in P['pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 result={'complete':False,'rows':[],'protocol_sha256':hashlib.sha256((HERE/'network_protocol.json').read_bytes()).hexdigest(),
  'prior_and_import_cpu_seconds':start}
 def record(x):
  with progress.open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n')
 def save():
  result['work_cpu_seconds']=time.process_time()-start;result['process_cpu_seconds']=time.process_time()
  out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 try:
  with mp.workdps(140):
   for case in P['cases']:
    record({'stage':'attempt','case':case});name,c,kind,xx=case;prefix=list(map(F,xx));b=min(prefix)
    target=sorted(range(2),key=lambda j:prefix[j],reverse=True)[1]
    Wi,Wh,Wo=model(c,kind);valid=partition(Wi,Wh,Wo,prefix,b,target)
    bound=candidate(Wi,Wh,Wo,prefix,b,target)
    # Independently derived toy recurrence, not the partition's root enumeration.
    S=c*c*prefix[0]+c*prefix[1];threshold=-S/2
    analytic=(cdf(b)-cdf(threshold))/cdf(b)
    if kind=='swapped':analytic=1-analytic
    if kind=='tied':analytic=mp.mpf(int(target==0))
    for edge in valid['boundaries']:
     x=F(edge['x'])
     expected=0 if kind=='tied' or x==threshold else int(x>threshold) if kind=='base' else int(x<threshold)
     assert edge['winner']==expected
    p=mp.mpf(valid['probability']);low=mp.mpf(bound['lower']);high=mp.mpf(bound['upper'])
    assert abs(p-analytic)<mp.mpf('1e-100')
    assert low<=p+mp.mpf('1e-100') and p<=high+mp.mpf('1e-100')
    if name=='vacuous':assert low==0 and high==1 and next(r for r in bound['rows'] if r['rival']==0)['status']=='vacuous_precheck'
    if kind=='tied':assert low==high==p
    if name=='strong':assert low>mp.mpf('.99')
    if name=='incorrect_anchor':assert high<mp.mpf('.01')
    row={'case':case,'target':target,'truncation':str(b),'weights':[[str(x) for x in Wi],[[str(x) for x in w] for w in Wh],[[str(x) for x in w] for w in Wo]],
     'validation':valid,'candidate':bound,'analytic_threshold':str(threshold),'analytic_probability':mp.nstr(analytic,110),'passed':True}
    result['rows'].append(row);record({'stage':'result','row':row});save()
   result['rational_strong_certificate']=rational_strong_certificate()
   assert len(result['rows'])==7
   result['candidate_forward_calls']=sum(r['candidate']['candidate_forward_calls'] for r in result['rows'])
   result['validation_forward_calls']=sum(r['validation']['validation_forward_calls'] for r in result['rows'])
   result['validation_cells']=sum(len(r['validation']['cells']) for r in result['rows'])
   result['validation_boundaries']=sum(len(r['validation']['boundaries']) for r in result['rows'])
   result['complete']=True;save();print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
 except BaseException as e:
  result['failure']=repr(e);record({'stage':'failure','error':repr(e)});save();raise
if __name__=='__main__':main()
