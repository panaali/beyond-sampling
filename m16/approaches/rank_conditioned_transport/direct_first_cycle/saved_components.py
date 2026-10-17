"""Read-only comparison of all prior terminal-mesh GL4 contribution rows."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ref=json.loads((HERE/'label_reference.json').read_text())
truth={str(r['label']):r['probability'] for r in ref['rows']}
source=HERE.parent/'panel_mesh_results.json'
out=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=[],scope='Saved-row diagnostic only; component norms are not challenge accuracy metrics.')
rows=[r for r in json.loads(source.read_text())['rows'] if r['stats']['order']==4]
for row in json.loads((HERE/'pilot.json').read_text())['rows']:
    if row['order']==4:rows.append(dict(**row,mode='direct' if row['direct'] else 'baseline'))
for row in rows:
    c=row['stats']['contributions'];errors=[c.get(str(j),0.)-truth[str(j)] for j in range(4)]
    out['rows'].append(dict(mode=row['mode'],estimate=row['estimate'],errors=errors,total=sum(errors),component_l1=sum(map(abs,errors)),component_l2=sum(e*e for e in errors)**.5))
(HERE/'saved_component_comparison.json').write_text(json.dumps(out,indent=2)+'\n')
