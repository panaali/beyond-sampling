"""Independent text-only verification; no imports from author or producer."""
import hashlib
import json
from pathlib import Path
import subprocess

base = Path('/workspace/work/recover-small_h2_56/review/shatter_prereg_review')
manifest = json.loads((base / 'manifest.json').read_text())
pins = dict(manifest['pins'])
for path, expected in pins.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
raw = (base / 'snapshot/README.md').read_bytes()
original = subprocess.check_output(['git', 'show', manifest['owner_commit'] + ':' + manifest['owner_path']])
assert raw == original
assert subprocess.check_output(['git', 'hash-object', '--stdin'], input=raw).decode().strip() == manifest['owner_git_blob']
tree = subprocess.check_output(['git', 'ls-tree', '--full-tree', '-r', '--name-only', manifest['owner_commit'], '--', 'm16/approaches/shatter_test/'])
assert tree.decode().splitlines() == [manifest['owner_path']]
assert tree == (base / 'AUTHOR_TREE.txt').read_bytes()
prior = json.loads(Path(next(p for p in pins if p.endswith('t242_complexity_review/source_review.json'))).read_text())
assert prior['complete'] and prior['pass_all'] and not prior['execution_authorized']
for name in ['manifest.json', 'provenance_completed.json']:
    pins[str(base / name)] = hashlib.sha256((base / name).read_bytes()).hexdigest()
here = Path(__file__).resolve().parent
for name in ['README.md', 'check_metadata.py']:
    pins[str(here / name)] = hashlib.sha256((here / name).read_bytes()).hexdigest()
out = dict(task='T254', reviewer='recover_34', complete=True, pass_all=True,
           execution_allowed=False, execution_authorized=False,
           owner_revision=manifest['owner_commit'], author_package=['441c7252', '34a6d277'],
           pins=pins, text_pins_verified=7, git_blobs_verified=1,
           owner_calls=0, arrays_decoded=0, scientific_numerical_calls=0,
           scope='Written source/theorem and text metadata only; no numerical or historical runtime validation')
print(json.dumps(out, indent=2))
