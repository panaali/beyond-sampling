"""Explain legacy-allocation differences without changing any saved comparison."""
import json,math
from fractions import Fraction as F
from pathlib import Path
from provenance import HERE,DATA,sha
def main():
 target=HERE/'historical_reconciliation.json';assert not target.exists()
 oldpath=DATA/'continuation_summary.json';newpath=HERE/'full_summary.json';old=json.loads(oldpath.read_text());new=json.loads(newpath.read_text());assert new['complete'];rows=[]
 for group in new['groups']:
  seed=group['seed'];prior=next(g for g in old['panels'] if g['seed']==seed and g['recipe']=='r85b');now=group['metrics']['historical']
  assert math.isclose(prior['statistics']['original72_ratio'],now['ratio_mean_budget'],rel_tol=2e-15)
  assert math.isclose(prior['per_copy_budget_ratio'],now['ratio_continuous_budget'],rel_tol=2e-15)
  ratios=[]
  for noise in range(20,24):
   for row in json.loads((DATA/f'k30_s{seed}_noise{noise}.json').read_text())['rows']:
    estimate=F(row['recipes']['r85b']['value']);truth=F(row['saved_truth']);cost=row['recipes']['r85b']['native_ops'];ratios.append((estimate-truth)**2*cost/(72*truth*(1-truth)))
  assert sum(r>=1 for r in ratios)==prior['copy_ratio_ge1']
  rows.append(dict(seed=seed,old_uniform_mean_budget=prior['statistics']['original72_ratio'],reconstructed_uniform_mean_budget=now['ratio_mean_budget'],old_continuous_matched=prior['per_copy_budget_ratio'],reconstructed_continuous_matched=now['ratio_continuous_budget'],new_integer_matched=now['ratio_integer_budget'],old_continuous_individual_losses=prior['copy_ratio_ge1'],new_integer_individual_losses=now['individual_losses']))
 out=dict(complete=True,rows=rows,pins={str(p):sha(p) for p in (oldpath,newpath,Path(__file__))},scope='Same original85B values/counters; mean allocation, continuous per-copy allocation and integer-floor per-copy allocation are different comparators. No repaired result, new estimator evaluation or new reference.')
 target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
