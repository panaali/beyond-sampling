"""Text/source pin binding only; no author, LP, model or array calls."""
import hashlib
import json
from pathlib import Path
import subprocess

B=Path('/workspace/work/recover-small_h2_78/review/gap_box_affine_range')
P=Path(__file__).resolve().parent
M=json.loads((B/'manifest.json').read_text())
pins=dict(M['pins'])
assert len(pins)==10
for path, digest in pins.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest, path
for name in ['PROOF.md','PLAN.md','README.md']:
    original=subprocess.check_output(['git','show','9ebdd477:review/gap_box_affine_range/'+name])
    assert original==(B/name).read_bytes()
assert subprocess.check_output(['git','show','93fc43c8:review/gap_box_affine_range/manifest.json'])==(B/'manifest.json').read_bytes()
pins[str(B/'manifest.json')]=hashlib.sha256((B/'manifest.json').read_bytes()).hexdigest()
for name in ['README.md','check_metadata.py']:
    pins[str(P/name)]=hashlib.sha256((P/name).read_bytes()).hexdigest()
print(json.dumps(dict(task='T256',reviewer='recover_34',complete=True,pass_all=True,
    execution_allowed=False,execution_authorized=False,
    author_commits=['9ebdd477','93fc43c8'],pins=pins,declared_pins_verified=10,
    exact_git_texts_verified=4,author_calls=0,lp_calls=0,array_decodes=0,
    scope='Written theorem/protocol and metadata only; future implementation/controls and application separately gated'),indent=2))
