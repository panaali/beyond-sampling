"""Preserve missing-rendered-report failure; contract existing raw records only."""
from pathlib import Path
import hashlib
P=Path(__file__).resolve().parent
p=P/'prune_tables.py';s=p.read_text()
assert hashlib.sha256(p.read_bytes()).hexdigest()=='ff869a922a75890dda08e28dbc8ddc845828887ffa13b7e42086a2e69e84d900'
old="report=(D/'prune_report.md').read_text();lines=[];groups={}"
new="assert not (D/'prune_report.md').exists(); report=None;lines=[];groups={}"
assert s.count(old)==1;s=s.replace(old,new)
assert s.count('assert line in report,line')==2
s=s.replace('assert line in report,line',"assert report is None  # no author-rendered table exists; do not infer parity")
old="'rows':lines,'pooled':pooled,"
new="'rows':lines,'pooled':pooled,'author_rendered_report_present':False,'scope_correction':'No rendered prune_report.md exists in pinned author tree; independent raw-table contraction only, no table-text parity claim',"
assert s.count(old)==1;s=s.replace(old,new)
for a,b in [('prune_tables_gate.json','prune_tables_recovery_gate.json'),('prune_tables_results.json','prune_tables_recovery_results.json'),('prune_tables_progress.jsonl','prune_tables_recovery_progress.jsonl')]:
 assert a in s;s=s.replace(a,b)
exec(compile(s,str(p),'exec'),{'__name__':'__main__','__file__':str(p)})
