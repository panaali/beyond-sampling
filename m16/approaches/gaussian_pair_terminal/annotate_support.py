"""Separate value-matched probes from genuinely consistent remaining rank."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=HERE/'interpolation_comparison.json';r=json.loads(p.read_text());selected=[]
for row in r['rows']:
    prefix=row['prefix'];j=row['label'];u=row['u'];tied=prefix[1-j]==u
    row['prefix_input_tie']=bool(row['jointly_consistent_target'] and tied)
    need=None if not row['jointly_consistent_target'] or tied else int(prefix[1-j]<u)
    row['rank_consistent_remaining_need']=need
    if need is not None:selected.append(dict(prefix=prefix,label=j,u=u,remaining_need=need,signed_error=row['signed_error'][need]))
r['rank_consistent_untied_subset']=selected;p.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(selected))
