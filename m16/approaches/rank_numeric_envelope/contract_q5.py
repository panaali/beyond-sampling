"""Contract actual Q5 histogram under the source envelope; all other kernels unknown."""
import hashlib,json,time
from pathlib import Path
from q5 import envelope
HERE=Path(__file__).resolve().parent;P=json.loads((HERE/'protocol.json').read_text());start=time.process_time()
p=next(Path(x) for x in P['pins'] if x.endswith('/rank_runtime_census/summary.json'));assert hashlib.sha256(p.read_bytes()).hexdigest()==P['pins'][str(p)]
d=json.loads(p.read_text());out={}
for name,mode in d['modes'].items():
    rows=[r for r in mode['rows'] if r['kernel']=='dlasq5'];totals=dict(arithmetic=0,negation=0,comparison=0)
    for row in rows:
        assert row['m'] in (3,4) and row['n'] in (0,1) and not row['query'];b=envelope(row['m'])
        for k,v in b.items():totals[k]+=row['calls']*v
    assert sum(r['calls'] for r in rows)==mode['kernel_calls']['dlasq5']
    out[name]=dict(q5_calls=sum(r['calls'] for r in rows),rows=rows,named_numeric_source_upper=totals,named_numeric_source_upper_total=sum(totals.values()),unpriced_kernel_calls={k:v for k,v in mode['kernel_calls'].items() if k!='dlasq5'},no_complete_rank_cost=True)
result=dict(modes=out,cpu_seconds=time.process_time()-start,qualification='Conditional pinned-source numeric upper subtotal using actual finite call counts. Not a full native/source/hardware cost. Integer control/index/memory and all other rank work remain explicit unknowns.')
(HERE/'q5_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))
