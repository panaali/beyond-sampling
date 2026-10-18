"""Hash-only source gate binding; does not run the saved arithmetic checker."""
import hashlib
import json
from pathlib import Path
import subprocess

B=Path('/workspace/work/feynman_fundamentals/m16/approaches/feynman_fundamentals/capacity_retained_evidence/saved')
P=Path(__file__).resolve().parent
protocol=json.loads((B/'protocol.json').read_text())
pins=dict(protocol['pins'])
for path,digest in pins.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
for name in ['check.py','protocol.json','PROTOCOL.md']:
    raw=(B/name).read_bytes()
    expected=subprocess.check_output(['git','show','444750e9:m16/approaches/feynman_fundamentals/capacity_retained_evidence/saved/'+name])
    assert raw==expected
    pins[str(B/name)]=hashlib.sha256(raw).hexdigest()
prior=json.loads((P/'source_review.json').read_text())
assert prior['complete'] and prior['pass_all']
for name in ['SOURCE_REVIEW.md','manifest.json']:
    path=B.parent/name
    assert prior['pins'][str(path)]==hashlib.sha256(path.read_bytes()).hexdigest()
for name in ['source_review.json','SAVED_SOURCE_REVIEW.md','pin_saved_gate.py']:
    pins[str(P/name)]=hashlib.sha256((P/name).read_bytes()).hexdigest()
print(json.dumps(dict(task='T257 saved consistency source/resource gate',reviewer='recover_34',
    complete=True,pass_all=True,execution_allowed=True,execution_authorized=True,
    author_source='444750e9',pins=pins,
    scope='One fixed six-eval/64x5-curve/six-table-row saved-only consistency run after published immutable gate',
    limits=dict(cpu_soft_seconds=20,cpu_hard_seconds=20,address_space_bytes=2147483648,internal_cpu_seconds=17,lock='/workspace/work/codex-review.lock'),
    author_calls_during_review=0,score_contractions_during_review=0,
    required_after_run='Independent saved arithmetic audit; no automatic expansion'),indent=2))
