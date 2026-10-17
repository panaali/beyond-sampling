"""Fixed15-group saved-data comparisons with original75-family truth boxes."""
import hashlib,json
from fractions import Fraction as Q
from pathlib import Path
from reference_compare import budget_samples
from tracer import HERE,ROOT
REVIEW=ROOT/'wider_audit.json'
def F(x):return Q(float(x))
def mean(xs):return sum(xs,Q(0))/len(xs)
def extrema(e,lo,hi):
 errlo=Q(0) if lo<=e<=hi else min((e-lo)**2,(e-hi)**2)
 errhi=max((e-lo)**2,(e-hi)**2)
 pqlo=min(lo*(1-lo),hi*(1-hi));pqhi=Q(1,4) if lo<=Q(1,2)<=hi else max(lo*(1-lo),hi*(1-hi))
 return errlo,errhi,pqlo,pqhi

def main():
 dst=HERE/'panel_comparison.json';assert not dst.exists()
 summary=json.loads((HERE/'panel_summary.json').read_text());saved=json.loads((ROOT/'frozen/ev_44.json').read_text());audit=json.loads(REVIEW.read_text());assert audit['family_size']==75 and audit['nominal_joint_coverage']==.95
 boxes={(r['seed'],r['index']):r for r in audit['mc_truth_boxes'] if r['H']==4};assert len(boxes)==25
 rows={(r['seed'],r['copy']):r for r in summary['rows']};assert len(rows)==25
 groups=[]
 for seed in range(5):
  for pop,inds in [('clean',[0]),('noise',list(range(1,5))),('pooled',list(range(5)))]:
   es=[];ps=[];costs=[];ext=[]
   for copy in inds:
    row=rows[seed,copy];old=saved[str(seed)]['rows'][copy];box=boxes[seed,copy]
    assert row['value']==old['h64'] and box['n']==10000000 and abs(old['mc']-box['hits']/box['n'])<2e-16
    es.append(F(row['value']));ps.append(Q(box['hits'],box['n']));costs.append(row['conditional']);ext.append(extrema(es[-1],F(box['lower']),F(box['upper'])))
   raw=mean([(e-p)**2 for e,p in zip(es,ps)]);pq=[p*(1-p) for p in ps]
   for sampler,per,ctl in [('dense_original',193,8),('analytic_zero_state_optimized',161,7)]:
    for allocation in ('uniform','matched'):
     budgets=[sum(costs)//len(costs)]*len(costs) if allocation=='uniform' else costs
     samples=[budget_samples(c,per,ctl) for c in budgets];ns=[x[0] for x in samples];spent=[x[1] for x in samples]
     assert sum(spent)<=sum(costs)
     variance=mean([v/n for v,n in zip(pq,ns)])
     blo=mean([x[0] for x in ext])/mean([x[3]/n for x,n in zip(ext,ns)])
     bhi=mean([x[1] for x in ext])/mean([x[2]/n for x,n in zip(ext,ns)])
     groups.append(dict(seed=seed,population=pop,sampler=sampler,allocation=allocation,copies=inds,candidate_source_budgets=costs,sampling_draws=ns,sampling_spent=spent,raw_mse=float(raw),sampling_variance=float(variance),raw_point_ratio=float(raw/variance),exact_raw_point_ratio=str(raw/variance),simultaneous_truth_box_ratio=[float(blo),float(bhi)],exact_box_ratio=[str(blo),str(bhi)]))
 out=dict(claim='3963b86c',groups=groups,summary_sha256=hashlib.sha256((HERE/'panel_summary.json').read_bytes()).hexdigest(),truth_audit_sha256=hashlib.sha256(REVIEW.read_bytes()).hexdigest(),truth_family=75,truth_nominal_joint_coverage=.95,noise_subtraction=False,h128_truth=False,scope='Fixed H4T4 saved panel, unchanged original Hopf with known generic mask defect. Conditional source allowances and numerically inverted CP truth boxes; no full runtime/binary/geometry certificate or wider challenge claim. Exact ratios use saved binary endpoints; interval construction is not outward-rounded.')
 dst.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'groups':len(groups),'optimized_matched_pooled':[{k:g[k] for k in ('seed','raw_point_ratio','simultaneous_truth_box_ratio')} for g in groups if g['population']=='pooled' and g['sampler']=='analytic_zero_state_optimized' and g['allocation']=='matched']}))
if __name__=='__main__':main()
