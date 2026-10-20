#!/usr/bin/env python3
"""Source-level operation inventory, not an instrumented full-FLOP replay."""
import json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
REV='a52da0d6a57d29aa2f76daa74c3360222f33aec2'
rows=[]
for i in range(15):
    r=json.loads(subprocess.check_output(['git','show',f'{REV}:m16/approaches/is_v4/est_{i}.json'],cwd=ROOT))
    pairs=r['hits']*r['n_components']
    # Direct sum of d*d triple products: 2*d*d multiplications, d*d-1 additions.
    contraction_flops=pairs*(3*10*10-1)
    rows.append(dict(index=i,hits=r['hits'],components=r['n_components'],hit_component_pairs=pairs,
                     direct_contraction_flops=contraction_flops,contraction_FPE=contraction_flops/5948,
                     published_C=r['C'],contraction_over_published_C=contraction_flops/5948/r['C']))
result=dict(source_revision=REV,rows=rows,scope='Stage2 dense quadratic contraction only, conventional direct evaluation. Does not reconstruct full charged cost or revised F.',
            mean_contraction_FPE=sum(r['contraction_FPE'] for r in rows)/15,
            sum_hit_component_pairs=sum(r['hit_component_pairs'] for r in rows),
            min_contraction_over_C=min(r['contraction_over_published_C'] for r in rows),
            max_contraction_over_C=max(r['contraction_over_published_C'] for r in rows))
Path(__file__).with_name('cost_inventory.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
