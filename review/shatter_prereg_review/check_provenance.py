"""Text/hash check only; no owner import or numerical work."""
import hashlib
import json
from pathlib import Path
import subprocess
here = Path(__file__).resolve().parent
m = json.loads((here/'manifest.json').read_text())
for path, sha in m['pins'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha, path
raw = subprocess.check_output(['git','show',m['owner_commit']+':'+m['owner_path']], cwd=here)
assert raw == (here/'snapshot/README.md').read_bytes()
assert subprocess.check_output(['git','rev-parse',m['owner_commit']+':'+m['owner_path']], cwd=here, text=True).strip() == m['owner_git_blob']
paths = subprocess.check_output(['git','ls-tree','--full-tree','-r','--name-only',m['owner_commit'],'m16/approaches/shatter_test'], cwd=here)
assert paths == (here/'AUTHOR_TREE.txt').read_bytes()
review = json.loads(Path(next(p for p in m['pins'] if p.endswith('t242_complexity_review/source_review.json'))).read_text())
assert review['complete'] and review['pass_all']
print(json.dumps({'complete':True,'pass_all':True,'text_pins':len(m['pins']),'owner_blobs':1,'numerical_calls':0,'execution_allowed':False}))
