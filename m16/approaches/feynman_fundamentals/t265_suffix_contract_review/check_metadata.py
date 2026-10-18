"""T265 text/Git metadata check only; no numerical kernel or payload reads."""
import hashlib,json,pathlib,subprocess
PEER=pathlib.Path('/workspace/work/recover-small_h2_78');B=PEER/'review/weak_suffix_success';O=pathlib.Path(__file__).resolve().parent
m=json.loads((B/'manifest.json').read_text());assert len(m['records'])==8
pins={};rows=[]
for r in m['records']:
 p=PEER/r['snapshot'];raw=p.read_bytes()
 expected=subprocess.check_output(['git','-C',str(PEER),'show',r['source_commit']+':'+r['source_path']])
 assert raw==expected and len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
 pins[str(p)]=r['sha256'];rows.append({'source':r['source_path'],'commit':r['source_commit'],'sha256':r['sha256'],'bytes':len(raw),'passed':True})
for n in ['THEORY.md','PLAN.md','SOURCE_INVENTORY.md','README.md','manifest.json']:
 p=B/n;raw=p.read_bytes();assert raw==subprocess.check_output(['git','-C',str(PEER),'show','4e2f2dbd:review/weak_suffix_success/'+n]);pins[str(p)]=hashlib.sha256(raw).hexdigest()
pins[str(O/'check_metadata.py')]=hashlib.sha256((O/'check_metadata.py').read_bytes()).hexdigest()
r={'complete':True,'pass_all':True,'execution_allowed':False,'task':'T265','assignment':'0e95e67d','producer_source':'4e2f2dbd','records':rows,'pins':pins,'calls':{'author_imports':0,'trained_payloads':0,'model':0,'cdf':0,'rng':0,'geometry':0},'scope':'Written conditional-law/domain/source contract only; not an implementation or trained numerical gate'}
assert not (O/'source_review.json').exists();(O/'source_review.json').write_text(json.dumps(r,indent=2)+'\n');print('eight exact source snapshots and five written files bound')
