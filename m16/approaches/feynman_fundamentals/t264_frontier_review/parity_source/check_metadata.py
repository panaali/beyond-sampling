"""Source metadata only; no imports of reviewed code or raw score contraction."""
import ast,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
B=Path('/workspace/work/recover-small_h2_56/review/pruning_cost_frontier/panel')
old=B.parent/'controls'
protocol=json.loads((B/'protocol.json').read_text())
pins=protocol['pins'].copy()
for f,h in pins.items(): assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
assert (B/'fixtures.json').read_bytes()==(old/'fixtures.json').read_bytes()
text=(old/'primitive.py').read_text()
replacements=[
 ("b=s=F();chosen=[];best=", "b=s=F();best_len=0;best="),
 ("for i in ids:\n            chosen.append(i);s+=c[i];b+=e[i]", "for prefix_len,i in enumerate(ids,1):\n            s+=c[i];b+=e[i]"),
 ("if ok and s>best['saved']:best={'indices':chosen.copy(),'saved':s,'error':b}", "if ok and s>best['saved']:best_len=prefix_len;best={'indices':[],'saved':s,'error':b}\n        best['indices']=ids[:best_len]")]
for a,b in replacements:
    assert text.count(a)==1
    text=text.replace(a,b)
assert text==(B/'primitive.py').read_text()
for name in ['primitive.py','run.py']: ast.parse((B/name).read_text())
for f in [B/n for n in ['primitive.py','run.py','fixtures.json','protocol.json','PROTOCOL.md']]+[P/'README.md',Path(__file__).resolve()]:
    pins[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
out=P/'source_gate.json'; assert not out.exists()
out.write_text(json.dumps({'complete':True,'pass_all':True,'execution_allowed':True,'task':'T264 storage parity','claim':'78a542ed','source_commit':'7d5856cd','source_publication':'f9e203ce','pins':pins,'scope':'One fixed12+6 new-helper parity run only, no retained target decoding','conditions':['Publish explicit execution/saved-review scope and immutable gate before calls','20/20CPU,2GiB/2GiB,25wall after reviewlock,internal17CPU,thread variables1','Refuse existing outputs/logs; preserve command/errors and any failure; no automatic retry','Independent saved parity required before separate retained-panel source/resource phase'],'metadata_checks':{'literal_rewrite':True,'fixture_bytes_equal':True,'protocol_hashes':len(protocol['pins'])},'reviewer_calls':{'primitive':0,'model':0,'target_payload':0,'mass':0,'rng':0}},indent=2)+'\n')
print('Metadata/source checks pass; no primitive calls.')
