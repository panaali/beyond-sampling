"""Text/Git metadata checks only, without scientific evaluation."""
import hashlib,json,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent
repo=Path('/workspace/work/recover-small_h2_78')
B=repo/'review/weak_suffix_success/reachable_corollary'
m=json.loads((B/'manifest.json').read_text());assert m['execution_allowed'] is False
pins={**m['own_pins'],**m['inherited_pins']}
for p,h in pins.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
for name in ['PROOF.md','README.md','manifest.json']:
    p=B/name
    original=subprocess.check_output(['git','show','c4d81aaf:'+str(p.relative_to(repo))],cwd=repo)
    assert original==p.read_bytes()
    pins[str(p)]=hashlib.sha256(original).hexdigest()
for p in [P/'README.md',Path(__file__).resolve()]:pins[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
out=P/'source_review.json';assert not out.exists()
out.write_text(json.dumps({'complete':True,'pass_all':True,'execution_allowed':False,'task':'T265 reachable same-summary written corollary','assignment':'7bce65ef','source_commit':'c4d81aaf','source_publication':'bbfb6f1e','pins':pins,'checks':['zero-start prefix reachability','identical strict top-two state and current target margin','all real last-input winner regions and first-index endpoints','exact correctness event and symbolic Gaussian difference','two-point minimax bound only','no almost-sure/population/trained/compression-family claim'],'metadata':{'own_git_files':3,'inherited_pins':4},'reviewer_scientific_calls':0},indent=2)+'\n')
print('Three Git text files and all inherited hashes pass; no scientific calls.')
