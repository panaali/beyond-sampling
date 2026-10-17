"""Summarize every frozen case, including incomplete and failed runs."""
import json
from pathlib import Path
HERE=Path(__file__).parent
rows=[];replays=[]
for mode in ('exact','recipe'):
    for kind in ('old','quadratic','nearest'):
        path=HERE/(kind+'_'+mode+'.json')
        if not path.exists():rows.append(dict(kind=kind,mode=mode,state='not_run'));continue
        x=json.loads(path.read_text());row={k:x.get(k) for k in ('kind','mode','complete','estimate','signed_reference_error','cpu_seconds','import_cpu','named_ops','families','completed_families','stats','failure','setup_cpu','setup_ops')}
        row['mass_capture']=x.get('mass_capture');rows.append(row)
for kind in ('quadratic','nearest'):
    for n in (2,4):
        p=HERE/(kind+'_mass_x'+str(n)+'.json')
        if p.exists():replays.append(json.loads(p.read_text()))
out=dict(rows=rows,mass_replays=replays,all_six_cases_complete=all(r.get('complete',False) for r in rows),reference=.477013125994119,reference_is_independent_conditional_numerical_reference=True,no_matched_accuracy_cost_claim=True,no_full_ladder_or_mse_claim=True)
(HERE/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:print(json.dumps(r))
