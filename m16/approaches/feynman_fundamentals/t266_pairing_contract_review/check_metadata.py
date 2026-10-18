"""Verify frozen text provenance; no scientific imports or execution."""
import hashlib,json,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent
repo=Path('/workspace/work/recover-small_h2_78');B=repo/'review/residual_pairing'
r=json.loads((B/'review_manifest.json').read_text());m=json.loads((B/'source_manifest.json').read_text())
assert r['execution_allowed'] is False and m['execution_allowed'] is False and len(m['records'])==6
pins=r['pins'].copy()
for f,h in pins.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
for item in m['records']:
    data=Path(item['snapshot']).read_bytes();assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
    original=subprocess.check_output(['git','show',item['commit']+':'+item['source']],cwd=repo)
    assert data==original
    pins[item['snapshot']]=item['sha256']
for name in ['PROOF.md','PLAN.md','SOURCE_INVENTORY.md','README.md','source_manifest.json','review_manifest.json']:
    f=B/name;assert subprocess.check_output(['git','show','1cdbc9be:'+str(f.relative_to(repo))],cwd=repo)==f.read_bytes()
    pins[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
for f in [P/'README.md',Path(__file__).resolve()]:pins[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
out=P/'source_review.json';assert not out.exists()
out.write_text(json.dumps({'complete':True,'pass_all':True,'execution_allowed':False,'task':'T266 written residual pairing contract','assignment':'9a5fe201','source_commit':'1cdbc9be','source_publication':'e827a54b','pins':pins,'checks':['unconditional reversal measure preservation','strict-rank label reversal and input/output ties','unchanged signed bias and exact opposite-sign absolute cancellation','normalized half-domain identity','fixed-cost sampling variance separate from quadrature','bounded source inventory and missing deployable surrogate/paired complex'],'metadata':{'source_git_snapshots':6,'own_git_documents':6},'reviewer_scientific_calls':0},indent=2)+'\n')
print('Frozen text/Git metadata pass; no scientific calls.')
