"""Saved-only summary; direct pooled means are retrospective diagnostics."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json
H=Path(__file__).resolve().parent
p=json.loads((H/'trained.json').read_text())
def frac(x):return F(*map(int,x))
def pack(x):return [str(x.numerator),str(x.denominator)]
rows=[]
for m in p['means']:
    u=m['unit'];selected=[r for r in p['rows'] if r['unit']==u];assert [r['sector'] for r in selected]==list(range(34));total=0.0
    for r in selected:total+=r['candidate']['base']['value']
    truth=frac(m['reference_exact_saved_sum']);e=abs(F(total)-truth);rel=e/abs(truth) if truth else None
    rows.append(dict(unit=u,direct_mean_hex=total.hex(),error=pack(e),relative_error=pack(rel) if rel is not None else None,absolute_pass=e<=F(1,10**10),relative_pass=rel<=F(1,10**8) if rel is not None else total==0,hybrid_error=m['diagnostics']['error']))
result=dict(cells=544,hybrid_cell_absolute_pass=p['cell_absolute_pass'],hybrid_cell_relative_pass=p['cell_relative_pass'],direct_cell_absolute_pass=p['baseline_absolute_pass'],direct_cell_relative_pass=p['baseline_relative_pass'],direct_pooled_absolute_pass=sum(r['absolute_pass'] for r in rows),direct_pooled_relative_pass=sum(r['relative_pass'] for r in rows),hybrid_pooled_absolute_pass=p['mean_absolute_pass'],hybrid_pooled_relative_pass=p['mean_relative_pass'],direct_max_pooled_absolute_error=float(max(frac(r['error']) for r in rows)),hybrid_max_pooled_absolute_error=float(max(frac(r['diagnostics']['error']) for r in p['means'])),hybrid_max_cell_absolute_error=float(max(frac(r['diagnostics']['error']) for r in p['rows'])),hybrid_max_cell_relative_error=float(max(frac(r['diagnostics']['relative_error']) for r in p['rows'] if r['diagnostics']['relative_error'] is not None)),fallbacks=p['mp80_evaluations'],fallback_fraction=p['mp80_evaluations']/544,rational_decodes=p['rational_decodes'],fallback_reasons=dict(Counter(r['candidate']['decision']['reason'] for r in p['rows'] if r['candidate']['mp_evaluations'])),cpu_seconds=p['cpu_seconds'],candidate_timers_sum=p['candidate_timers_sum'],partial_float_counts=p['direct_named_partial_counts'],inherited_preparation_counts=p['inherited_preparation_counts'],retrospective_direct_mean_rows=rows,scope='Saved-only retrospective direct pooled diagnostic: fallback repairs cell-relative errors, but is not shown necessary for this one-model pooled tolerance. No new candidate/MP/reference/MC calls, no general guarantee or full cost/speed comparison.')
out=H/'summary.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['direct_pooled_absolute_pass','direct_pooled_relative_pass','direct_max_pooled_absolute_error','hybrid_max_pooled_absolute_error','fallbacks']}))
