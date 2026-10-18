"""Text/hash check only. No author imports or binary/result payload reads."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())
records = []
for row in manifest['source_files']:
    raw = (root / row['snapshot']).read_bytes()
    assert len(raw) == row['bytes']
    assert hashlib.sha256(raw).hexdigest() == row['sha256']
    original = subprocess.check_output(['git', 'show', manifest['author_revision'] + ':' + row['source_path']])
    assert original == raw
    assert subprocess.check_output(['git', 'hash-object', '--stdin'], input=raw).decode().strip() == row['git_blob']
    if row['snapshot'].endswith('.py'):
        ast.parse(raw)
    records.append(row['source_path'])
assert hashlib.sha256((root / 'AUTHOR_TREE.txt').read_bytes()).hexdigest() == manifest['tree_sha256']
for name, digest in manifest['prior_pins'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
print(json.dumps({'text_provenance_pass': True, 'files': records,
                  'author_calls': 0, 'binary_payload_reads': 0,
                  'execution_authorized': False}, indent=2))
