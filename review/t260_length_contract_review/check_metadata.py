"""Verify text and Git metadata only; never load author code or binary payloads."""
import hashlib,json,subprocess
from pathlib import Path
base=Path('/workspace/work/recover-small_h4_4');folder=base/'review/formula_description_length'
def git(*a): return subprocess.check_output(['git','-C',str(base),*a])
def sha(b): return hashlib.sha256(b).hexdigest()
m=json.loads((folder/'manifest.json').read_text());records=[]
for r in m['files']:
    p=folder/r['snapshot'];b=p.read_bytes();obj=f"{r['revision']}:{r['path']}"
    assert b==git('show',obj) and sha(b)==r['sha256'] and len(b)==r['size']
    assert git('rev-parse',obj).decode().strip()==r['blob']
    records.append({'path':str(p),'sha256':sha(b),'bytes':len(b),'git_blob':r['blob']})
assert len(records)==15
pins={}
for name in ['CONTRACT.md','SOURCE_INVENTORY.md','README.md','manifest.json','AUTHOR_TREE.txt']:
    p=folder/name;b=p.read_bytes();assert b==git('show',f'bbb8446c:{p.relative_to(base)}');pins[str(p)]=sha(b)
assert (folder/'AUTHOR_TREE.txt').read_bytes()==git('ls-tree','-rl','--full-tree',m['owner_revision'],'--','m16/approaches/neuron_account')
r={'task':'T260','reviewer':'recover_34','complete':True,'pass_all':True,'execution_allowed':False,'execution_authorized':False,'producer_commit':'bbb8446c','delegation':'3c52f71b','pins':pins,'source_records':records,'written_checks':['comparable final-hidden versus logit versus full-state reconstruction targets','final-step all-coordinate overwrite and transitive hinge-bank dependency closure','IDs selected columns pooled fallback casts and conditional coefficient formula','nonserialization scoped to inspected source and tracked tree metadata','T25 framed prefix grammar and conditional versus standalone accounting','prospective fixed compression and held-out paired validation prerequisites'],'excluded':['author imports or codec execution','binary checkpoint/array decode','fit/model/score/RNG calls','historical numerical parity','actual trained coefficient totals or rate curves','minimality or complexity lower bounds']}
Path(__file__).with_name('source_review.json').write_text(json.dumps(r,indent=2)+'\n')
print('PASS: 15 exact text sources, five package files, exact owner tree metadata; zero author calls')
