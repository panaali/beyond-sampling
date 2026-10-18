"""Saved exact bound diagnostics, no numerical kernel calls."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json
H=Path(__file__).resolve().parent
z=json.loads((H/'results.json').read_text())
rows=[];counts=Counter()
for r in z['rows']:
    v=r['result'];counts.update(v['counts'])
    rows.append(dict(fixture=r['fixture'],order=r['order'],status=v['status'],upper_float=float(F(*map(int,v['probability_upper']))) if v['status']=='bounded' else None,below_threshold=r['below_existing_1e_10_diagnostic']))
result=dict(cases=len(rows),bounded=sum(r['status']=='bounded' for r in rows),below_threshold=sum(r['below_threshold'] for r in rows),rows=rows,loose_fixtures=sorted({r['fixture'] for r in rows if not r['below_threshold']}),bound_named_categories=dict(counts),cpu_seconds=z['cpu_seconds'],qualification='Bounds isolate exact-real source constants only. Loose upper is not measured numerical error. Full source evaluation floating error remains excluded.')
(H/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
