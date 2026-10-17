import hashlib,json
from pathlib import Path
HERE=Path(__file__).parent
old=json.loads((HERE/'controls.json').read_text());results=[]
for name,sha in old['source_hashes'].items():
 p=Path(name);now=hashlib.sha256(p.read_bytes()).hexdigest();results.append(dict(path=name,initial_sha256=sha,final_sha256=now,unchanged=sha==now))
for name,sha in old['checkpoint_hashes'].items():
 p=Path('/workspace/work/t114_checkpoints')/name;now=hashlib.sha256(p.read_bytes()).hexdigest();results.append(dict(path=str(p),initial_sha256=sha,final_sha256=now,unchanged=sha==now))
assert all(r['unchanged'] for r in results)
(HERE/'source_bookends.json').write_text(json.dumps(dict(results=results),indent=2)+'\n')
print('Unchanged source/checkpoint hashes:',len(results))
