"""Independent saved-only scalar audit; no author import or quadrature replay."""
import hashlib,json,time,traceback
from pathlib import Path
import mpmath as mp
HERE=Path(__file__).resolve().parent
AUTHOR=Path('/workspace/work/recover-small_h2_78/m16/approaches/conditional_order_bounds')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cdf(x):return (1+mp.erf(x/mp.sqrt(2)))/2
def density(x):return mp.exp(-x*x/2)/mp.sqrt(2*mp.pi)
def independent(b,a,s):
 z=cdf(b)
 if a>b:return mp.exp(s*a+s*s/2)*cdf(b+s)/z,a+density(b)/z
 left=mp.exp(s*a+s*s/2)*cdf(a+s)
 right=mp.exp(-s*a+s*s/2)*(cdf(b-s)-cdf(a-s))
 below=a*cdf(a)+density(a)
 above=density(a)-density(b)-a*(cdf(b)-cdf(a))
 return (left+right)/z,(below+above)/z

def main():
 start=time.process_time();dst=HERE/'t189_scalar_review.json';assert not dst.exists()
 files=['scalar.py','controls.py','THEORY.md','protocol.json','controls.json','controls_progress.jsonl','controls1.log']
 out=dict(complete=False,assignment='d3dbee50',author_source='5fae242f',author_raw='f6060637',source_gate='T55 message clearance, recorded3b9aa608; original protocol text is historical',pins={str(AUTHOR/n):sha(AUTHOR/n) for n in files},review_source_sha256=sha(Path(__file__)),rows=[],new_author_calls=0,new_quadrature_calls=0,scope='Saved output/domain coverage plus independent erf-formula160-digit contraction; finite precision, no outward certificate or independent runtime domain execution.')
 def save():out['cpu']=time.process_time()-start;tmp=dst.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(dst)
 save()
 try:
  P=json.loads((AUTHOR/'protocol.json').read_text());R=json.loads((AUTHOR/'controls.json').read_text());events=[json.loads(l) for l in (AUTHOR/'controls_progress.jsonl').read_text().splitlines()]
  assert sha(AUTHOR/'protocol.json')=='e74aa01690ca8c97e998a4ef67e48534c04461a60900bf07e6fef505eee737f2'==R['protocol_sha256']
  for p,h in P['pins'].items():assert sha(Path(p))==h
  expected=[['0','0','0'],['0','0','1'],['1','0','0.5'],['-1','-2','0.75'],['-1','0','1'],['2','3','2'],['-8','-8','0.25'],['8','-1','1.5']]
  assert P['cases']==expected and R['complete'] and len(R['rows'])==8 and len(R['domain'])==4 and len(events)==20
  assert P['precheck_rates']==['0','0.01','0.1','1']
  mp.mp.dps=160
  for i,(case,row) in enumerate(zip(expected,R['rows'])):
   assert row['case']==case and row['passed'] and events[2*i]==dict(stage='attempt',case=case) and events[2*i+1]==dict(stage='result',row=row)
   b,a,s=map(mp.mpf,case);calc=independent(b,a,s);hi=list(map(mp.mpf,row['value120']));lo=list(map(mp.mpf,row['value80']));ref=list(map(mp.mpf,row['reference100']));err=[abs(v-w) for v,w in zip(hi,ref)]
   assert all(abs(v-w)<=mp.mpf('1e-115')*(1+abs(v)) for v,w in zip(calc,hi))
   assert all(e<=mp.mpf('1e-65')*(1+abs(v)) for e,v in zip(err,hi))
   assert all(abs(v-w)<=mp.mpf('1e-65')*(1+abs(v)) for v,w in zip(hi,lo))
   assert all(abs(e-mp.mpf(saved))<mp.mpf('1e-135') for e,saved in zip(err,row['errors']))
   mean=calc[1];assert mean>0
   high=mean+mp.mpf(1)/4;low=mean-mp.mpf(1)/4;t=mp.mpf('.0001')
   bound=mp.exp(min(mp.mpf(0),-t*high+mp.log(independent(b,a,t)[0])));assert bound<1
   assert abs(high-mp.mpf(row['margin_above_mean']))<mp.mpf('1e-105')
   assert abs(bound-mp.mpf(row['bound_fixed_rate']))<mp.mpf('1e-105')
   for rate,saved in zip(P['precheck_rates'],row['below_mean_log_values']):
    t=mp.mpf(rate);v=-t*low+mp.log(independent(b,a,t)[0]);assert v>=-mp.mpf('1e-150') and abs(v-mp.mpf(saved))<mp.mpf('1e-105')
   out['rows'].append(dict(case=case,independent=[mp.nstr(v,155) for v in calc],formula_differences=[mp.nstr(abs(v-w),12) for v,w in zip(calc,hi)],saved_quadrature_discrepancies=[mp.nstr(e,12) for e in err],passed=True));save()
  assert [r['name'] for r in R['domain']]==P['domain']==['negative_rate','nonfinite_bound','negative_weight','zero_allowance']
  for i,row in enumerate(R['domain']):assert events[16+i]==dict(stage='domain',row=row)
  assert all(r['status']=='rejected' for r in R['domain'][:3])
  z=R['domain'][3];assert z['margins']==[-1,0,1] and list(map(mp.mpf,z['bounds']))==[1,1,0] and z['status']=='passed'
  assert 0<R['cpu_seconds']<20 and R['process_cpu_seconds']>=R['cpu_seconds']
  out.update(complete=True,controls=8,domain_records=4,progress_records=20,author_cpu=R['cpu_seconds'],domain_limitation='Runtime rejection was checked only in author saved records; reviewer inspected source guards without executing them.');save()
  print(json.dumps({k:out[k] for k in ('complete','controls','domain_records','cpu')}))
 except Exception as exc:out['failure']=dict(error=repr(exc),traceback=traceback.format_exc());save();raise
if __name__=='__main__':main()
