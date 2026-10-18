"""Frozen source-text provenance only; no module or scientific payload imports."""
import hashlib,json,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;repo=Path('/workspace/work/recover-small_h2_78');B=repo/'review/arc_a_target'
m=json.loads((B/'manifest.json').read_text());assert m['task']=='T273' and len(m['files'])==9
pins={}
for item in m['files']:
    f=repo/item['snapshot'];data=f.read_bytes();assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
    assert subprocess.check_output(['git','show',item['source_commit']+':'+item['source_path']],cwd=repo)==data
    pins[str(f)]=item['sha256']
for name in ['RECIPE_DECISION.md','README.md','manifest.json']:
    f=B/name;assert subprocess.check_output(['git','show','128f8287:'+str(f.relative_to(repo))],cwd=repo)==f.read_bytes()
    pins[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
for f in [P/'README.md',Path(__file__).resolve()]:pins[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
out=P/'source_review.json';assert not out.exists()
out.write_text(json.dumps({'complete':True,'pass_all':True,'execution_allowed':False,'task':'T273 recipe-domain written review','assignment':'7c14443b','source_commit':'128f8287','source_publication':'ac79035b','pins':pins,'source_files':9,'checks':['literal H4T3 hardcoded shape/region construction','shared latent q=.2 versus canonical frozen-gate surrogate','rank_exact import-time sampled setup and Sobol volume approximation','fixed-budget and classification decisions remain explicit','missing generic extension is not executed failure or family negative'],'scientific_calls':0,'sealed_truth_reads':0},indent=2)+'\n')
print('Nine exact source Git texts verified; no scientific calls.')
