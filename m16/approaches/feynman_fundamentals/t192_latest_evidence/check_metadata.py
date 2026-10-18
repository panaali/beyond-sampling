"""T255 metadata/schema check only; no model or saved numerical contraction."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
P=Path(__file__).resolve().parent;R=P.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((P/'manifest.json').read_text());assert len(m['files'])==20
for row in m['files']:
 b=(P/row['snapshot']).read_bytes();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
 assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==row['git_blob']
 assert b==subprocess.check_output(['git','show',m['author_head']+':'+row['source_path']],cwd=R)
 if row['snapshot'].endswith('.py'):ast.parse(b)
assert sha(P/'AUTHOR_TREE.txt')==m['tree_sha256']
rows=[json.loads(l) for l in (P/'snapshot/results/mc_target/blocks.jsonl').read_text().splitlines()]
assert len(rows)==300
assert all(set(r)=={'block','seed','n','correct','correct_float32','sec'} for r in rows)
assert all(type(r['block']) is int and type(r['correct']) is int and type(r['n']) is int for r in rows)
k6=json.loads((P/'snapshot/results/prefix/k6_masses.json').read_text());assert k6['complete'] is False
r=dict(task='T255',complete=True,pass_all=True,meaning='twenty frozen blob bindings and JSON schema only; not numerical totals or source-theorem independent clearance',source_files=20,mc_record_count=300,mc_arithmetic_verified=False,k6_complete=False,author_calls=0,binary_payload_reads=0,execution_authorized=False,pins={str(f):sha(f) for f in [P/'manifest.json',P/'SOURCE_REVIEW.md',P/'README.md',Path(__file__)]})
assert not (P/'metadata_result.json').exists();(P/'metadata_result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ['task','complete','source_files','mc_record_count','mc_arithmetic_verified']}))
