"""Aggregate existing branch flags; no trained mass evaluation."""
import json,time
from collections import Counter
import numpy as np
from inventory import HERE,AUTHOR,PINS
start=time.process_time();controls=json.loads((HERE/'controls.json').read_text());branch={r['name']:r for r in controls['rows']}
counts=Counter();seen=[]
for batch in range(8):
    stem=AUTHOR/'plackett_rule_doubling'/f'batch_{batch}';meta=json.loads(stem.with_suffix('.json').read_text());assert meta['complete']
    assert meta['source_hashes'][str(AUTHOR/'plackett_rule_doubling/run.py')]==PINS[AUTHOR/'plackett_rule_doubling/run.py']
    data=np.load(stem.with_suffix('.npz'));seen.extend(data['indices'].tolist())
    for thin,cut in zip(data['thin'],data['cutoff']):counts['cutoff' if cut else 'thin' if thin else 'wide']+=1
assert seen==list(range(35098))
legacy=sum(counts[name]*row['legacy'] for name,row in branch.items());assert legacy==404438808
categories=Counter();opaque=Counter();native=Counter()
for name,row in branch.items():
    for key,value in row['ledger']['totals'].items():categories[key]+=counts[name]*value
    for item in row['ledger']['opaque_calls']:opaque[tuple(item['key'])]+=counts[name]*item['count']
    for item in row['ledger']['native_calls']:native[tuple(item['key'])]+=counts[name]*item['count']
result=dict(counts=dict(counts),legacy_partial=legacy,inclusive_explicit=sum(categories.values()),categories=dict(categories),opaque=[dict(key=list(k),count=v) for k,v in opaque.items()],native=[dict(key=list(k),count=v) for k,v in native.items()],cpu_seconds=time.process_time()-start,complete_implementation_cost=False)
(HERE/'mass_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('categories','opaque','native')}))
