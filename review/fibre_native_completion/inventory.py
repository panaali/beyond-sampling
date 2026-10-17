"""Freeze existing source and data before the unchanged component continuation."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=Path('/workspace/work/recover-small_h4_4/m16/approaches')
NATIVE=BASE/'fibre_t4_native_cost';PRIOR=BASE/'fibre_t4_recovery'
paths=[NATIVE/x for x in ('tracer.py','inventory.json','component.json','summary.json','COLD_SCOPE.md','HANDOFF.md')]
paths += [BASE/'plackett_cost_review/inventory.py']
paths += [PRIOR/x for x in ('noise_identity.json','noise_preflight.json','noise_summary.json')]
paths += sorted((PRIOR/'noise_components').glob('*.json'))
inv=json.loads((NATIVE/'inventory.json').read_text());paths += [Path(x['path']) for x in inv['files']]
assert len(list((PRIOR/'noise_components').glob('*.json')))==47
out={'claim':'bb3a2778','native':str(NATIVE),'prior':str(PRIOR),'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'indices':list(range(48)),'reuse_index':0,'source_review':'ba401fec','component_limit_cpu':110,'model_sha256':json.loads((PRIOR/'noise_preflight.json').read_text())['model_sha256']}
p=HERE/'inventory.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n')
