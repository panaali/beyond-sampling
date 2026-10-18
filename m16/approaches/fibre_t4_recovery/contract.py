"""Exact stored-float arithmetic for recovered data; independent new references only."""
from fractions import Fraction as F
from pathlib import Path
import hashlib,json,time,math
HERE=Path(__file__).resolve().parent;start=time.process_time();inventory=json.loads((HERE/'inventory.json').read_text());SRC=Path(inventory['source_root'])
for x in inventory['files']:assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']
for x in inventory['references']:assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']
data=json.loads((SRC/'results_H8_T4.json').read_text())['rows'];rows=[];ops1=320;explicit_sampler=512
ref0=json.loads(Path(inventory['references'][0]['path']).read_text());ref1=json.loads(Path(inventory['references'][1]['path']).read_text())
refs={'seed0':(ref0['reference_accuracy'],ref0['conditional_reference_integration_bound']), 'seed1':(ref1['exact']['accuracy_positive'],ref1['exact']['conditional_bound'])}
for name,r in data.items():
 p=F(r['truth']);se=F(r['truth_se']);e=F(r['F2L3']['est'])-p;ops=F(r['F2L3']['ops']);v=p*(1-p)
 row=dict(name=name,estimate=r['F2L3']['est'],mc_truth=r['truth'],mc_se=r['truth_se'],inferred_mc_n=float(v/(se*se)),
  error_against_mc=float(e),error_in_mc_se=float(e/se),legacy_ops=int(ops),reported_seconds=r['F2L3']['sec'],
  legacy_mc_variance_subtracted_ratio=float(max(e*e-se*se,0)*ops/(ops1*v)),
  traces=r['F2L3']['traces'],pieces=r['F2L3']['pieces'],sampled_pattern_lower_bound=r['patterns'],face_cells=r['face_cells'],
  face_over_sampled_patterns_upper_bound_on_face_over_true_regions=float(F(r['face_cells'],r['patterns'])))
 if name in refs:
  ref,bound=map(F,refs[name]);err=abs(F(r['F2L3']['est'])-ref);low=ref-bound;high=ref+bound
  assert 0<low<high<1 and low>F(1,2) and err>bound
  varlo=high*(1-high);varhi=low*(1-low)
  # Arithmetic headroom if true p lies in the conditional reference interval.
  budgetlo=explicit_sampler*varlo/(err+bound)**2;budgethi=explicit_sampler*varhi/(err-bound)**2
  row['corrected_reference']=dict(value=float(ref),conditional_integration_bound=float(bound),signed_error=float(F(r['F2L3']['est'])-ref),
   conditional_abs_error_interval=[float(err-bound),float(err+bound)],
   legacy_count_ratio_using_legacy320_sampler=float(err*err*ops/(ops1*ref*(1-ref))),
   conditional_break_even_cost_explicit512_sampler=[float(budgetlo),float(budgethi)],
   mixed_legacy_estimator_count_divided_by_explicit_budget=float(ops/budgetlo),
   comparison_scope='Saved intended seed identity; current source matches result snapshot, but original row has no checkpoint/source execution hashes. Reference bound excludes floating geometry/roundoff. Legacy estimator count is incomplete and does not share the explicit512 convention; mixed ratio is not a win.')
 rows.append(row)
mean=lambda xs:sum(xs,F())/len(xs)
mse=max(mean([(F(r['F2L3']['est'])-F(r['truth']))**2-F(r['truth_se'])**2 for r in data.values()]),F())
meanops=mean([F(r['F2L3']['ops']) for r in data.values()]);meanvar=mean([F(r['truth'])*(1-F(r['truth'])) for r in data.values()])
out=dict(rows=rows,legacy_group_ratio=float(mse*meanops/(ops1*meanvar)),legacy_sampler=ops1,independent_explicit_sampler=explicit_sampler,noise_rows_present=0,
 total_historical_estimator_seconds=sum(r['F2L3']['sec'] for r in data.values()),cpu_seconds=time.process_time()-start,
 limitations=['MC variance subtraction/floor is an estimator of squared bias, not zero-error evidence.','Only two independently corrected numerical references are available here.','Face cells/sample patterns upper-bounds face cells/true regions; cannot prove a lower complexity separation.','No full cost or noisy-panel pass; checkpoint execution provenance remains partial.'])
(HERE/'contraction.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
