"""Independent Decimal raw-row budget arithmetic and equality-defect probe."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from decimal import Decimal as Q,localcontext
import importlib.util,json,time
start=time.process_time();H=Path(__file__).parent;P=Path('/workspace/work/recover-small_h2_910/review/recipe112_recovery');A=json.loads((P/'aggregate.json').read_text());DATA=P/'snapshot/out';groups=rowscount=0;core=[]
def q(x):return Q.from_float(x) if isinstance(x,float) else Q(x)
def mean(v):return sum(v,Q(0))/len(v)
def match(x,y):assert abs(float(x)-y)<2e-12*max(1,abs(y))
with localcontext() as ctx:
 ctx.prec=65
 for report in A['files']:
  raw=json.loads((DATA/report['file']).read_text());h=raw['H'];sampler=q(3*h*(h+1)+3*h);rs=[]
  for g in report['groups']:
   original=raw['zoo'][str(g['zoo'])];rows=original['rows'][1:];assert len(rows)==raw['n']*len(raw['seeds']);ps=[q(r['truth']) for r in rows];err=[q(r['est'])-q(r['truth']) for r in rows];op=[q(r['ops']) for r in rows];mse=mean([v*v for v in err]);variance=[p*(1-p) for p in ps];legacy=mse/(mean(variance)*sampler/mean(op));individual=mse/mean([v*sampler/o for v,o in zip(variance,op)]);match(legacy,g['legacy_ratio']);match(individual,g['individual_budget_ratio']);match(legacy,original['ratio']);match(mse,g['mean_squared_error']);match(mean(err),original['bias']);match(mean(op),original['ops']);rs.append(float(legacy));groups+=1;rowscount+=len(rows)
  if raw['H'] in (4,5,6,8) and raw['mode']=='abs' and raw['sigma']==.05 and raw['seeds']==[80,81,82,83]:core.append(dict(H=h,models=sum(g['noisy_models'] for g in report['groups']),worst_legacy_ratio=max(rs)))
 count=json.loads((DATA/'count.json').read_text());counter_mismatches=[]
 for g in A['count']:
  r=count[str(g['H'])];v=(sum(q(x['auto']) for x in r['rows'])/q(r['sample_auto']))/(sum(q(x['charge']) for x in r['rows'])/q(r['sample_conv']));match(v,g['aggregate_multiplier']);counter_mismatches.append(dict(H=g['H'],maximum_recorded_value_mismatch=max(abs(x['diff']) for x in r['rows'])))
assert groups==A['total_groups']==165 and rowscount==A['total_noisy_rows']==10720
spec=importlib.util.spec_from_file_location('snapshot_counter',P/'snapshot/rec112.py');R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R);assert '__eq__' not in R.Num.__dict__ and '__ne__' not in R.Num.__dict__;assert (R.Num(0.)==0) is False and (R.Num(0.)!=0) is True
sem=json.loads((P/'semantics.json').read_text());h8=next(r for r in sem['models'] if r['H']==8);assert h8['original_count']==h8['saved']['auto'] and h8['equality_repaired']==h8['float_reimplementation']==h8['recipe'] and h8['original_wrapped']!=h8['recipe'];assert abs(h8['original_wrapped']-h8['recipe'])==h8['saved']['diff']
# All four analytic networks select first label for every input: hidden is nonnegative,
# and readout rows are all equal or (h,0,-h). Continuous iid ranks give label0 prob1/3.
for r in sem['analytic']:assert r['analytic_truth']==1/3
out=dict(raw_groups=groups,noisy_rows=rowscount,core=sorted(core,key=lambda r:r['H']),count_panel_value_mismatches=counter_mismatches,numeric_zero_identity_bug_reproduced=True,h8_equality_only_value_restored=True,h8_original_count=h8['original_count'],h8_patched_count=h8['equality_repaired_count'],analytic_truth_all_four=1/3,review_cpu=time.process_time()-start,scope='Independent raw arithmetic and tiny original-wrapper probe, no trained estimator rerun. Legacy ratios are historical conventions; counter transfer is unvalidated because wrapped execution changes values. Positive-volume tie failures do not alone invalidate all generic noisy-model truths.')
(H/'recipe112_recovery_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
