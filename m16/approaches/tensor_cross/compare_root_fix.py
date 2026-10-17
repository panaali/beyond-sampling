"""Compare frozen experiment before/after shared walker correction, no model calls."""
import json
from pathlib import Path
p=Path(__file__).resolve().parent
old=json.loads((p/'dev_results_before_root_fix.json').read_text())
new=json.loads((p/'dev_results.json').read_text())
records=[]
for a,b in zip(old['records'],new['records'],strict=True):
    keys=('model','order','rank','sweeps')
    assert all(a[k]==b[k] for k in keys)
    assert len(a['detail'])==len(b['detail'])
    row={k:b[k] for k in keys}
    row.update(accuracy_before=a['accuracy'],accuracy_after=b['accuracy'],
        accuracy_change=b['accuracy']-a['accuracy'],faces_before=a['faces'],faces_after=b['faces'],
        cost_before=a['cost_fpe'],cost_after=b['cost_fpe'],cost_ratio=b['cost_fpe']/a['cost_fpe'],
        cpu_before=a['cpu_seconds'],cpu_after=b['cpu_seconds'])
    differences=[]
    for c,d in zip(a['detail'],b['detail'],strict=True):
        assert c['kind']==d['kind']
        assert c['faces']==d['faces'] and c['cells']==d['cells'] and c['pivots']==d['pivots']
        for k in ('integral','weighted_error'):
            differences.append(abs(c[k]-d[k]))
        for k in ('true_mean','predicted_mean','relative_l2','top5_true_mass_share','top5_abs_error_mass','negative_fraction'):
            differences.append(abs(c['audit'][k]-d['audit'][k]))
    row['max_numerical_audit_or_integral_difference']=max(differences)
    records.append(row)
result=dict(corrected_backend_commit='ae05eef',settings_changed=False,
            max_absolute_accuracy_difference=max(abs(r['accuracy_change']) for r in records),
            max_numerical_audit_or_integral_difference=max(r['max_numerical_audit_or_integral_difference'] for r in records),
            total_construction_faces=sum(r['faces'] for r in new['records']),
            total_construction_fpe=sum(r['cost_fpe'] for r in new['records']),
            total_audit_faces=sum(d['audit']['new_faces'] for r in new['records'] for d in r['detail']),
            total_audit_fpe=sum(d['audit']['walk_fpe'] for r in new['records'] for d in r['detail']),
            corrected_experiment_cpu_seconds=new['cpu_seconds'],records=records)
(p/'root_fix_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
