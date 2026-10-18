"""Frozen source and tree-inventory metadata only."""
import hashlib,json,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;repo=Path('/workspace/work/recover-small_h2_910');B=repo/'review/t272_band_review'
m=json.loads((B/'manifest.json').read_text());assert len(m['files'])==9
pins={}
for item in m['files']:
    f=repo/item['local_path'];data=f.read_bytes();assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
    assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item['git_blob']
    assert subprocess.check_output(['git','show',item['source_commit']+':'+item['source_path']],cwd=repo)==data
    pins[str(f)]=item['sha256']
assert hashlib.sha256((B/'retained_inventory.txt').read_bytes()).hexdigest()==m['retained_inventory_sha256']
for name in ['SOURCE_REVIEW.md','README.md','manifest.json','retained_inventory.txt']:
    f=B/name;assert subprocess.check_output(['git','show','690a79a1:'+str(f.relative_to(repo))],cwd=repo)==f.read_bytes()
    pins[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
for f in [P/'README.md',Path(__file__).resolve()]:pins[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
out=P/'source_review.json';assert not out.exists()
out.write_text(json.dumps({'complete':True,'pass_all':True,'execution_allowed':False,'task':'T275 bounded T272 source interpretation','assignment':'905fec76','source_commit':'690a79a1','source_publication':'eec1c85a','pins':pins,'source_git_texts':9,'checks':['F0c additive extension and scalar overlap witness','sampled signature transitions versus analytic nonsmoothness','same seed707 across F0a aggregate rules; covariance sign unknown','preregistered secondary/finite diagnostic scope','numerical window/cap/tie/runtime/cost limits','no numerical result reversal or broad-family negative'],'scientific_calls':0,'result_payload_decodes':0},indent=2)+'\n')
print('Nine Git text payloads and fixed inventory verified; no scientific calls.')
