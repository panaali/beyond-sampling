"""Only fresh-reference substitutions in the independently reviewed60-comparison code."""
import hashlib,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parent;COST=HERE.parent/'hopf_native_cost';sys.path.insert(0,str(COST))
base=COST/'panel_compare.py'
PIN_HASH='e77c80ccf0a81ca5ccf47b8ddc7fbea0578f18a6e25113867e69c2cd1c773f44'
assert hashlib.sha256(base.read_bytes()).hexdigest()==PIN_HASH
src=base.read_text()
changes=[
 ('from tracer import HERE,ROOT','from tracer import ROOT\nHERE=Path(__file__).resolve().parent\nCOST=HERE.parent/"hopf_native_cost"',1),
 ("REVIEW=ROOT/'wider_audit.json'","REVIEW=HERE/'reference_summary.json'",1),
 ("HERE/'panel_comparison.json'","HERE/'fresh_comparison.json'",1),
 ("HERE/'panel_summary.json'","COST/'panel_summary.json'",2),
 ("ROOT/'frozen/ev_44.json'","HERE/'fresh_ev_44.json'",1),
 ("audit['family_size']==75","audit['family_size']==25",1),
 ("box['n']==10000000","box['n']==100000000",1),
 ("claim='3963b86c'","claim='c78c0999'",1),
 ('truth_family=75','truth_family=25',1),
 ('Fixed H4T4 saved panel, unchanged original Hopf with known generic mask defect.','Fresh fixed100M reference per model for unchanged H4T4 Hopf with known generic mask defect.',1),
]
for old,new,n in changes:
 assert src.count(old)==n,(old,src.count(old),n)
 src=src.replace(old,new)
module=types.ModuleType('fixed_fresh_reference_comparison');module.__file__=str(HERE/'compare.py')
exec(compile(src,str(base)+' [fresh fixed reference only]','exec'),module.__dict__)
if __name__=='__main__':module.main()
