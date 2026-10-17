"""Independent bookkeeping of saved same-state terminal diagnostics."""
import hashlib
import json
from pathlib import Path

base=Path('/workspace/work/recover-small_h4_4/m16/approaches/rank_conditioned_transport/reachable_cycle')
path=base/'strata.json'
data=json.loads(path.read_text())
rows=[]
for row in data['rows']:
    defects={key:sum(c[key] for c in row['strata'].values())-row[key]
             for key in ('signed','absolute','squared')}
    mass=sum(c['mass'] for c in row['strata'].values())
    assert max(map(abs,defects.values()))<2e-15
    assert abs(mass-.75)<2e-15
    rows.append(dict(hidden_order=row['hidden_order'],mode=row['mode'],
                     mass=mass,defects=defects,
                     origin_absolute=row['strata']['origin']['absolute'],
                     axis1_absolute=row['strata']['axis1']['absolute'],
                     total_absolute=row['absolute'],signed=row['signed']))
out=dict(rows=rows,stratum_mass=data['stratum_mass'],
         total_tie_mass=sum(data['stratum_zero_tie_mass'].values()),
         distinct_nonzero_mask_slopes=data['no_other_positive_volume_ties_by_distinct_affine_slopes'],
         source_hashes={f:hashlib.sha256((base/f).read_bytes()).hexdigest()
                        for f in ('diagnostic.py','strata.json')},
         scope='Saved-result arithmetic and source law review; no new engine quadrature or error enclosure.')
Path(__file__).with_name('reachable_summary_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
