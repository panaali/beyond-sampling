"""Text/hash provenance only; no author imports or numerical payloads."""
import hashlib, json, pathlib, subprocess
p = pathlib.Path(__file__).resolve().parent
m = json.loads((p/'manifest.json').read_text())
rows=[]
for r in m['records']:
    b=(p/r['snapshot']).read_bytes()
    original=subprocess.check_output(['git','show',m['owner_commit']+':'+r['source']], cwd=p)
    blob=subprocess.check_output(['git','rev-parse',m['owner_commit']+':'+r['source']], cwd=p,text=True).strip()
    assert b==original and len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'] and blob==r['blob']
    rows.append({'source':r['source'],'passed':True})
out=p/'source_checks.json'
assert not out.exists()
out.write_text(json.dumps({'complete':True,'pass_all':True,'rows':rows,'author_calls':0,'scientific_numeric_calls':0},indent=2)+'\n')
