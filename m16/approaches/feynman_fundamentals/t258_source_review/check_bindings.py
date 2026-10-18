"""Metadata-only independent T258 Git/text binding check; no author imports."""
import hashlib,json,pathlib,subprocess
BASE=pathlib.Path('/workspace/work/recover-small_h2_56/review/replacement_claims_recheck')
OUT=pathlib.Path(__file__).resolve().parent
m=json.loads((BASE/'manifest.json').read_text())
assert len(m['records'])==10
rows=[]
def git(*args):
    return subprocess.check_output(['git','-C',str(BASE),*args])
for r in m['records']:
    payload=(BASE/r['snapshot']).read_bytes()
    original=git('show',m['owner_commit']+':'+r['source'])
    blob=git('rev-parse',m['owner_commit']+':'+r['source']).decode().strip()
    assert payload==original and blob==r['blob']
    assert len(payload)==r['bytes'] and hashlib.sha256(payload).hexdigest()==r['sha256']
    rows.append(dict(source=r['source'],blob=blob,sha256=r['sha256'],bytes=len(payload),passed=True))
tree=git('ls-tree','--full-tree','-rl',m['owner_commit'],'--','m16/approaches/neuron_account/')
assert tree==(BASE/'AUTHOR_TREE.txt').read_bytes()
pins={str(BASE/p):hashlib.sha256((BASE/p).read_bytes()).hexdigest() for p in ['manifest.json','SOURCE_REVIEW.md','README.md','source_checks.json','AUTHOR_TREE.txt']+[r['snapshot'] for r in m['records']]}
pins[str(OUT/'check_bindings.py')]=hashlib.sha256((OUT/'check_bindings.py').read_bytes()).hexdigest()
result=dict(complete=True,pass_all=True,execution_allowed=False,task='T258',scope='Written source/theorem and exact text/Git provenance only',owner_commit=m['owner_commit'],producer_note_commit='b526dcf9',assignment='cf4861e2',rows=rows,tree_entries=len(tree.splitlines()),pins=pins,calls=dict(author=0,model=0,fit=0,rng=0,numeric_archives=0,score_contractions=0),limitations=['No empirical score validation or reproduction','No historical runtime/model/archive identity certification','No new execution authorization','No numerical payload or binary checkpoint reads'])
assert not (OUT/'source_review.json').exists()
(OUT/'source_review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'complete':True,'pass_all':True,'text_blobs':len(rows),'tree_entries':len(tree.splitlines())}))
