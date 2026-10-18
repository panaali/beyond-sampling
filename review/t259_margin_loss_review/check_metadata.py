"""Text/Git provenance only; no author imports, arrays or numerical controls."""
import hashlib,json,subprocess
from pathlib import Path
base=Path('/workspace/work/recover-small_h2_910')
folder=base/'review/margin_loss_protocol'
commit='3bc9e2c3'
def git(*args):
    return subprocess.check_output(['git','-C',str(base),*args])
def sha(b): return hashlib.sha256(b).hexdigest()
m=json.loads((folder/'manifest.json').read_text())
records=[]
for row in m['files']:
    b=(base/row['path']).read_bytes()
    obj=f"{row['source_commit']}:{row['source_path']}"
    assert b==git('show',obj)
    assert len(b)==row['bytes'] and sha(b)==row['sha256']
    assert git('rev-parse',obj).decode().strip()==row['git_blob']
    records.append({'path':str(base/row['path']),'sha256':sha(b),'bytes':len(b),'git_blob':row['git_blob']})
pins={}
for name in ['PROOF.md','SOURCE_REVIEW.md','PROTOCOL.md','README.md','manifest.json']:
    p=folder/name;b=p.read_bytes();rel=str(p.relative_to(base))
    assert b==git('show',f'{commit}:{rel}')
    if name!='manifest.json': assert sha(b)==m['owned_text_pins'][rel]
    pins[str(p)]=sha(b)
result={'task':'T259','reviewer':'recover_34','complete':True,'pass_all':True,'execution_allowed':False,'execution_authorized':False,'producer_commit':commit,'claim':'92a67233','scope':'Independent written algebra/source/provenance review only','pins':pins,'source_records':records,'written_checks':['common output-linear ridge/readout map with common routing, feature bank and solver convention','unordered and ordered pair sum/mean factors','matched penalty and centered/common-mode orthogonal decomposition','singular/intercept/target-refitting/finite-arithmetic limits','CellLinear design-derived penalties and fallback; unclipped phaseC_diag post-ReLU targets','phaseC2 selected families and ReLU remain outside equality theorem'],'excluded':['author imports','arrays/checkpoints/models','fit/forward/RNG calls','numerical controls or empirical score review','99 percent impossibility or feature information ceiling','full runtime or cost verification']}
Path(__file__).with_name('source_review.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS: five author text blobs and five written package files; zero author calls')
