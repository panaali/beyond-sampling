"""Fixed60 H5 group and50 individual saved-data comparisons, original truth boxes."""
import hashlib,json,resource,time
from fractions import Fraction as F
from pathlib import Path
from compare import budget,ratio,encode
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mean(xs):return sum(xs,F(0))/len(xs)
def extrema(e,lo,hi):
 emin=F(0) if lo<=e<=hi else min((e-lo)**2,(e-hi)**2);emax=max((e-lo)**2,(e-hi)**2)
 pmin=min(lo*(1-lo),hi*(1-hi));pmax=F(1,4) if lo<=F(1,2)<=hi else max(lo*(1-lo),hi*(1-hi))
 return emin,emax,pmin,pmax

def main():
 resource.setrlimit(resource.RLIMIT_CPU,(19,20));dst=HERE/'panel_comparison.json';assert not dst.exists();start=time.process_time()
 out={'complete':False,'groups':[],'individuals':[],'source_sha256':sha(__file__)}
 def save():dst.write_text(json.dumps(out,indent=2)+'\n')
 save()
 try:
  scope=json.loads((HERE/'panel_compare_scope.json').read_text())
  for p,h in scope['pins'].items():assert sha(p)==h,p
  summary=json.loads((HERE/'panel_summary.json').read_text());assert summary['complete'] and summary['source_sha256']==sha(HERE/'panel_summary.py')
  assert len(summary['rows'])==25;rows={(r['seed'],r['copy']):r for r in summary['rows']};assert set(rows)=={(s,c) for s in range(5) for c in range(5)}
  audit=json.loads(Path(scope['truth_box_file']).read_text());assert audit['family_size']==75 and audit['nominal_joint_coverage']==.95
  boxes={(r['seed'],r['index']):r for r in audit['mc_truth_boxes'] if r['H']==5 and r['T']==4};assert set(boxes)==set(rows)
  models=json.loads((HERE/'models.json').read_text());saved={(r['seed'],r['copy']):r for r in models['models']}
  for key,r in rows.items():
   b=boxes[key];m=saved[key];assert r['model_sha256']==m['model_sha256'] and r['value_hex']==m['saved_h64_hex']
   assert b['n']==m['mc_n'] and b['hits']==m['mc_count_inferred_from_report']
   e=F(r['value']);p=F(b['hits'],b['n']);lo,hi=F(b['lower']),F(b['upper']);assert 0<lo<p<hi<1 and 0<=e<=1
   for name,per,ctl in [('dense_original',281,8),('analytic_initial_zero',231,7)]:
    n,spent=budget(r['conditional'],per,ctl);ends=[ratio(n,e,q) for q in (lo,hi)]
    out['individuals'].append(dict(seed=key[0],copy=key[1],sampler=name,candidate_budget=r['conditional'],draws=n,spent=spent,point_ratio=encode(ratio(n,e,p)),nominal_truth_box_ratio_range=[encode(F(0) if lo<=e<=hi else min(ends)),encode(max(ends))]));save()
  for seed in range(5):
   for pop,copies in [('clean',[0]),('noise',[1,2,3,4]),('pooled',[0,1,2,3,4])]:
    es=[F(rows[seed,c]['value']) for c in copies];ps=[F(boxes[seed,c]['hits'],boxes[seed,c]['n']) for c in copies];costs=[rows[seed,c]['conditional'] for c in copies]
    ext=[extrema(e,F(boxes[seed,c]['lower']),F(boxes[seed,c]['upper'])) for e,c in zip(es,copies)];raw=mean([(e-p)**2 for e,p in zip(es,ps)])
    for name,per,ctl in [('dense_original',281,8),('analytic_initial_zero',231,7)]:
     for alloc in ('uniform','matched'):
      budgets=[sum(costs)//len(costs)]*len(costs) if alloc=='uniform' else costs
      ns_spent=[budget(C,per,ctl) for C in budgets];ns=[x[0] for x in ns_spent];spent=[x[1] for x in ns_spent];assert sum(spent)<=sum(costs)
      variance=mean([p*(1-p)/n for p,n in zip(ps,ns)])
      lo=mean([x[0] for x in ext])/mean([x[3]/n for x,n in zip(ext,ns)]);hi=mean([x[1] for x in ext])/mean([x[2]/n for x,n in zip(ext,ns)])
      out['groups'].append(dict(seed=seed,population=pop,copies=copies,sampler=name,allocation=alloc,candidate_budgets=costs,draws=ns,spent=spent,raw_mse=encode(raw),sampling_variance=encode(variance),point_ratio=encode(raw/variance),nominal_truth_box_ratio_range=[encode(lo),encode(hi)]));save()
  assert len(out['groups'])==60 and len(out['individuals'])==50
  out.update(complete=True,summary_sha256=sha(HERE/'panel_summary.json'),truth_box_file_sha256=sha(scope['truth_box_file']),family_size=75,nominal_joint_coverage=.95,numerical_beta_not_outward_rounded=audit['numerical_beta_not_outward_rounded'],noise_subtraction=False,h128_truth=False,scope='All fixed H5 original models. Conditional source costs and original nominal75-family10M CP boxes only. Free true labels/RNG for sampler; no fullnative/runtime, corrected-mask or general-rung claim. No comparison selection, newtruth, retuning or model calls.')
 except BaseException as e:out['failure']=repr(e);raise
 finally:out['cpu_seconds']=time.process_time()-start;save()
 print(json.dumps({'complete':out['complete'],'groups':len(out['groups']),'individuals':len(out['individuals'])}))
if __name__=='__main__':main()
