"""Fixed-panel summaries with both MC allocation conventions; no tuning."""
from pathlib import Path
import json
import numpy as np
HERE=Path(__file__).parent

def stats(rows,name):
 rows=[r for r in rows if name in r];n=len(rows)
 if n==0:return None
 p=np.array([r['truth'] for r in rows]);e=np.array([r[name]['value']-r['truth'] for r in rows]);pv=p*(1-p)
 if name=='t75':
  budget=np.array([r[name]['own_ops']/r[name]['sample_ops'] for r in rows]);claim=np.array([r[name]['claimed_ops']/72 for r in rows])
 else:
  budget=np.array([r[name]['strict_ops']/72 for r in rows]);claim=budget.copy()
 def calc(ix,b):return float(np.mean(e[ix]**2)/np.mean(pv[ix]/b[ix]))
 ids=np.arange(n);direct=calc(ids,budget);uniform=float(np.mean(e*e)*np.mean(budget)/np.mean(pv))
 rng=np.random.default_rng(991);boot=[calc(rng.integers(n,size=n),budget) for _ in range(1000)]
 return dict(n=n,mse=float(np.mean(e*e)),bias=float(np.mean(e)),max_abs_error=float(np.max(abs(e))),mean_accuracy=float(np.mean(p)),mean_sample_equivalents=float(np.mean(budget)),per_model_allocation_ratio=direct,uniform_mean_cost_ratio=uniform,claimed_counter_per_model_ratio=calc(ids,claim),claimed_counter_uniform_ratio=float(np.mean(e*e)*np.mean(claim)/np.mean(pv)),bootstrap_ci=np.quantile(boot,[.025,.975]).tolist(),individual_failures=int(np.sum(e*e>=pv/budget)),max_individual_ratio=float(np.max(e*e/(pv/budget))))

def main():
 rows=[]
 for path in sorted(HERE.glob('*_k*_s*.json')):
  d=json.loads(path.read_text())
  if not d.get('complete'):continue
  row=dict(file=path.name,k=d['k'],seed=d['training_seed'],kind=d['kind'],sigma=d['sigma'],cpu=d['cpu'],t75=stats(d['rows'],'t75'),fibre=stats(d['rows'],'fibre'),clean_t75=stats([d['clean']],'t75'),clean_fibre=stats([d['clean']],'fibre'),max_reference_error=max(abs(r['reference_difference']) for r in d['rows']),max_recipe_difference=max(abs(r['t75']['value']-r['t75']['own_value']) for r in d['rows']),failed_checks=sum(not all(r['checks'].values()) for r in d['rows']))
  rows.append(row)
 out=dict(cells=rows,n_completed_cells=len(rows),total_model_rows=sum(r['t75']['n']+1 for r in rows),recorded_phase_cpu=sum(r['cpu'] for r in rows),cost_caveat='T75 independent scalar recount; F2L3 reexecuted prior R4 strict count, not a new complete instrumentation audit. Bootstrap CIs are pointwise noisy-copy CIs, not simultaneous or training-seed generalization.')
 (HERE/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
 for r in rows:print(r['file'],'T75',round(r['t75']['per_model_allocation_ratio'],5),r['t75']['bootstrap_ci'],'F2L3',round(r['fibre']['per_model_allocation_ratio'],5),'checks',r['failed_checks'])
if __name__=='__main__':main()
