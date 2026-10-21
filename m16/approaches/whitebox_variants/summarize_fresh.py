import json
from pathlib import Path
H=Path(__file__).resolve().parent
ours=json.loads((H/'results/fresh_scores.json').read_text())
base={r['row']:r for r in json.loads((H.parent/'whitebox_stronger/results/fresh_scores.json').read_text())}
for r in ours:
 b=base[r['row']];ratio=r['adjusted']/b['adjusted_error'];r.update(baseline_adjusted=b['adjusted_error'],ratio_to_e30=ratio,strict_point_pass=ratio<=1,lenient_point_pass=ratio<=10)
groups={}
for method in ('e30_rank448','e30_tau30'):
 rs=[r for r in ours if r['method']==method];ratio=sum(r['adjusted'] for r in rs)/sum(r['baseline_adjusted'] for r in rs)
 groups[method]={'n':len(rs),'mean_adjusted':sum(r['adjusted'] for r in rs)/len(rs),'ratio_to_e30':ratio,'strict_point_pass':ratio<=1,'lenient_point_pass':ratio<=10}
cpu=sum(json.loads((H/'results'/f"{r['method']}_{r['row']}.json").read_text())['worker_cpu_seconds'] for r in ours)
report=dict(rows=ours,groups=groups,worker_cpu_seconds=cpu,scope='Frozen two-public-model check untouched by our tuning; upstream authors may have used data. Comparator is frozen e30, not AlgZoo F; point bars only, no population confidence claim.')
(H/'results/fresh_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
