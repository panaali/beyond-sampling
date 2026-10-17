"""Contract saved full-corpus rows only; no candidate/reference evaluation."""
import hashlib,json,math
from pathlib import Path
P=Path(__file__).resolve().parent
batches=[json.loads((P/f'full_{i}.json').read_text()) for i in range(8)]
assert all(b['complete'] for b in batches)
rows=[r for b in batches for r in b['rows']];assert len(rows)==35098 and {r['index'] for r in rows}==set(range(35098))
assert sum(r['reused_from_panel'] for r in rows)==256
assert all('failure' not in r and r['parity'] for r in rows)
keys=['negative_principal_minors','zero_principal_minors','product_floor_hits','denominator_floor_hits','pc_clip_hits','pc_nonfinite','scalar_3det_calls','scalar_4det_calls','inverse_calls']
s=dict(cases=len(rows),reused256_once=256,source_mass=math.fsum(r['value'] for r in rows),positive_mass=math.fsum(r['reference'] for r in rows),old_mass=math.fsum(r['old_source'] for r in rows),signed_error=math.fsum(r['difference'] for r in rows),L1_error=math.fsum(abs(r['difference']) for r in rows),max_abs_error=max(abs(r['difference']) for r in rows),max_relative_error=max(abs(r['difference'])/r['reference'] for r in rows if r['reference']>0),reference_bound_sum=math.fsum(r['reference_bound'] for r in rows),reference_only_bound_exceedances=sum(abs(r['difference'])>r['reference_bound'] for r in rows),gate_exceedances=sum(abs(r['difference'])>1e-10+r['reference_bound'] for r in rows),negative_values=sum(r['value']<0 for r in rows),cutoffs=sum(r['diagnostics']['cutoff'] for r in rows),source_cpu_once=math.fsum(r['source_cpu'] for r in rows),observer_cpu_once=math.fsum(r['diagnostic_cpu'] for r in rows),legacy_ops_once=sum(r['legacy_ops'] for r in rows),shard_process_cpu=math.fsum(b['cpu'] for b in batches))
s.update({k:sum(r['diagnostics'].get(k,0) for r in rows) for k in keys})
s['source_accuracy']=1-s['source_mass'];s['positive_accuracy']=1-s['positive_mass'];s['accuracy_difference']=s['source_accuracy']-s['positive_accuracy']
s['old_L1_error']=math.fsum(abs(r['old_difference']) for r in rows);s['old_max_abs_error']=max(abs(r['old_difference']) for r in rows)
s['worst_abs10']=[{k:r[k] for k in ['index','value','reference','difference','reference_bound']} for r in sorted(rows,key=lambda r:abs(r['difference']),reverse=True)[:10]]
s['worst_relative10']=[{k:r[k] for k in ['index','value','reference','difference','reference_bound']} for r in sorted(rows,key=lambda r:abs(r['difference'])/r['reference'] if r['reference']>0 else -1,reverse=True)[:10]]
s['pins']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for pat in ['full_[0-7].json','full_nodes_*.npz','panel.json','full_protocol.json','full_controls.json','full.py'] for f in P.glob(pat)}
s['scope']='Finite same-ray source evaluation; reused256 once, ordinary floating, conditional reference truncation bounds only. No full native-cost or universal accuracy guarantee. Source and observer timed separately; fullcontrol/setup/storage/import/hash are separate review overhead.'
(P/'full_summary.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in s.items() if k not in ['pins','worst_abs10','worst_relative10']}))
