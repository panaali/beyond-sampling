"""Written-contract metadata only, with inherited saved payload hash-only."""
import hashlib,json,subprocess
from pathlib import Path
B=Path('/workspace/work/recover-small_h2_78/review/gap_box_affine_range/enclosure_contract')
P=Path(__file__).resolve().parent
manifest=json.loads((B/'manifest.json').read_text());pins=dict(manifest['pins'])
assert len(pins)==9 and manifest['execution_authorized'] is False
for path,digest in pins.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
for name in ['PROOF.md','PLAN.md','README.md','manifest.json']:
    raw=(B/name).read_bytes()
    assert raw==subprocess.check_output(['git','show','98f6f8e4:review/gap_box_affine_range/enclosure_contract/'+name])
    pins[str(B/name)]=hashlib.sha256(raw).hexdigest()
for name in ['README.md','check_metadata.py']:pins[str(P/name)]=hashlib.sha256((P/name).read_bytes()).hexdigest()
print(json.dumps(dict(task='T256 written enclosure integration',reviewer='recover_34',
    complete=True,pass_all=True,execution_allowed=False,execution_authorized=False,
    author_commit='98f6f8e4',pins=pins,declared_pins_verified=9,git_texts_verified=4,
    author_calls=0,numerical_calls=0,array_decodes=0,
    scope='Written exact-real representation, propagation, fallback and control-plan only'),indent=2))
