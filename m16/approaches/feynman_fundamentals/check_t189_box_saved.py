"""Independent rational CDF enclosure and all saved whole-box predicates."""
from pathlib import Path
from fractions import Fraction as F
from math import factorial,isqrt
import hashlib,json,sys,time
sys.set_int_max_str_digits(0)
H=Path(__file__).resolve().parent;P=Path('/workspace/work/recover-small_h2_78/m16/approaches/conditional_order_bounds/box_mass')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def iv(d):return F(d['lower']),F(d['upper'])
def atanbounds(n):
 x=F(1,n);s=sum(((-1)**j*x**(2*j+1)/F(2*j+1) for j in range(80)),F(0));return s,s+x**161/161

def owncdf(x):
 if x==0:return F(1,2),F(1,2)
 if x<0:
  l,u=owncdf(-x);return 1-u,1-l
 l5,u5=atanbounds(5);l239,u239=atanbounds(239);pl,pu=16*l5-4*u239,16*u5-4*l239
 Q=2**400;lo=F(isqrt((2*pl*Q*Q).numerator//(2*pl*Q*Q).denominator),Q);up=F(isqrt((2*pu*Q*Q).numerator//(2*pu*Q*Q).denominator)+1,Q)
 assert lo*lo<=2*pl<=2*pu<=up*up
 lower=sum((F((-1)**j*x**(2*j+1),2**j*factorial(j)*(2*j+1)) for j in range(32)),F(0))
 upper=lower+F(x**65,2**32*factorial(32)*65)
 return F(1,2)+lower/up,F(1,2)+upper/lo

def main():
 start=time.process_time();pr=json.load(open(P/'protocol.json'));rp=json.load(open(P/'recovery_protocol.json'));old=json.load(open(P/'results.json'));new=json.load(open(P/'recovery_results.json'))
 for obj in (pr,rp):
  for f,h in obj['pins'].items():assert sha(f)==h,f
 assert new['complete'] and new['saved_pair_checks']==72 and new['recovery_protocol_sha256']==sha(P/'recovery_protocol.json')
 assert not old['complete'] and old['models']==[] and old['failure']=="TypeError('both arguments should be Rational instances')"
 assert new['reused_control_rows']==old['control_rows'] and new['reused_mass']==old['per_ordered_label_box_mass']
 assert new['preserved_first_attempt_cpu_seconds']==old['process_cpu_seconds'] and new['preserved_first_attempt_accounting']==old['accounting']
 for k in ('network_forwards','checkpoint_loads','reference_quadrature_calls','random_draws','repeated_controls','repeated_model_checks'):assert new[k]==0
 assert len(old['control_rows'])==9
 for row,case in zip(old['control_rows'][:4],pr['predicate_controls'],strict=True):
  name,L,i,j,expected=case;om=[k for k in range(4) if k not in (i,j)];S=sum(L[k] for k in om);A=2*L[i]+L[j];r=row['result'];assert row['passed'] and row['name']==name and row['expected']==expected==(S>0 and A<=F(S,4))
  assert list(map(F,r['L_by_input']))==L and r['omitted']==om and F(r['anchor_margin_abs_upper'])==A and F(r['allowance_mean_strict_lower'])==F(S,4)
 cdf={}
 for row,x in zip(old['control_rows'][4:],[-2,-1,0,1,2],strict=True):
  assert row['kind']=='cdf' and F(row['x'])==x and row['passed'];a,b=iv(row['enclosure']);lo,hi=owncdf(x)
  assert 0<=a<=lo<=hi<=b<=1 and b-a<F(1,10**25),(x,float(a-lo),float(hi-b));cdf[x]=(a,b)
 p0,p1,p2=cdf[0],cdf[1],cdf[2]
 # Independent positive interval contraction with the exact saved endpoints.
 ml=(p2[0]-p1[1])*(p1[0]**3-p0[1]**3)/3;mu=(p2[1]-p1[0])*(p1[1]**3-p0[0]**3)/3
 assert iv(new['reused_mass'])==(ml,mu) and ml>0
 data=json.load(open(pr['trained_data']));assert [m['name'] for m in new['models']]==[m['name'] for m in data['models']]
 summaries=[];checks=0
 for output,model in zip(new['models'],data['models'],strict=True):
  bounds={}
  for age in model['envelope']['rows']:
   for pair in age['pairs']:bounds.setdefault((pair['a'],pair['b']),[None]*4)[3-age['age']]=F(*map(int,pair['bound']))
  assert output['checkpoint_sha256']==model['checkpoint_sha256'] and output['reused_global_envelope_setup_counts']==model['envelope']['counts']
  assert [(b['largest'],b['second']) for b in output['boxes']]==[(i,j) for i in range(4) for j in range(4) if i!=j]
  passed=[]
  for box in output['boxes']:
   i,j=box['largest'],box['second'];om=[k for k in range(4) if k not in (i,j)];rs=[k for k in range(4) if k!=j];assert [r['rival'] for r in box['rivals']]==rs
   outcomes=[]
   for row,rival in zip(box['rivals'],rs,strict=True):
    L=bounds[(max(j,rival),min(j,rival))];A=2*L[i]+L[j];S=sum(L[k] for k in om);q=S/4;yes=S>0 and A<=q
    assert list(map(F,row['L_by_input']))==L and row['omitted']==om and row['largest']==i and row['second']==j
    assert F(row['anchor_margin_abs_upper'])==A and F(row['allowance_mean_strict_lower'])==q and row['positive_omitted_allowance']==(S>0) and row['certified_vacuous']==yes
    assert (None if row['retained_to_threshold_ratio'] is None else F(row['retained_to_threshold_ratio']))==(A/q if S else None)
    outcomes.append(yes);checks+=1
   yes=all(outcomes);assert box['certified_whole_box_vacuous']==yes and iv(box['certified_undecided_mass'])==((ml,mu) if yes else (F(0),F(0)))
   if yes:passed.append([i,j]);assert box['uncertified_status'] is None
  n=len(passed);assert output['certified_boxes']==n and iv(output['certified_union_mass'])==(n*ml,n*mu) and F(output['integrated_accuracy_interval_width_lower'])==n*ml
  w=n*ml;N=(1/(4*w*w)).numerator//(1/(4*w*w)).denominator+1 if n else None
  assert (output['sampling_SE_threshold']['samples_at_or_above'] if n else output['sampling_SE_threshold'])==N
  if n:assert 4*N*w*w>1 and 4*(N-1)*w*w<=1
  summaries.append({'name':output['name'],'certified_boxes':passed,'width_lower_display':float(w),'width_upper_display':float(n*mu),'max_Bernoulli_SE_sample_threshold':N})
 assert checks==72
 ev=[json.loads(x) for x in (P/'recovery_progress.jsonl').read_text().splitlines()];assert len(ev)==52
 assert [r['row'] for r in ev if r['stage']=='model_result']==new['models']
 assert [r['box'] for r in ev if r['stage']=='box_result']==[b for m in new['models'] for b in m['boxes']]
 assert new['total_attempt_cpu_seconds']==new['process_cpu_seconds']+old['process_cpu_seconds']
 r={'complete':True,'owner_artifact_commit':'21925681','raw_sha256':sha(P/'recovery_results.json'),'initial_failure_sha256':sha(P/'results.json'),'recovery_protocol_sha256':sha(P/'recovery_protocol.json'),'checker_sha256':sha(Path(__file__)),'independent_rational_cdf_controls':5,'predicate_controls':4,'exact_saved_pair_checks':72,'models':summaries,'reused_controls_once':9,'candidate_forward_quadrature_calls_in_review':0,'author_total_attempt_cpu':new['total_attempt_cpu_seconds'],'review_cpu':time.process_time()-start,'qualification':'This proves a lower bound on the width of this zero-anchor/global-absolute-L Chernoff family over disjoint top-two boxes. It does not lower-bound actual prediction error/MSE or exclude tighter conditional methods. Sampling thresholds compare width to maximal standard error, not matched computation.'}
 (H/'t189_box_saved_review.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ('complete','models','review_cpu')}))
if __name__=='__main__':main()
