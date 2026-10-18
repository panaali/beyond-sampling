"""Independent read-only source bindings. Never imports author modules or binary data."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
P=Path('/workspace/work/recover-small_h2_34/review/neuron_intervention_source')
HERE=Path(__file__).resolve().parent
R=HERE.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((P/'manifest.json').read_text());assert m['task']=='T251' and m['reviewer_delegation']=='3a63641c'
assert len(m['source_files'])==5
names=[];pins={}
for row in m['source_files']:
    path=P/row['snapshot'];b=path.read_bytes();assert len(b)==row['bytes'] and sha(path)==row['sha256']
    assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==row['git_blob']
    assert subprocess.check_output(['git','show',m['author_revision']+':'+row['source_path']],cwd=R)==b
    if path.suffix=='.py':ast.parse(b)
    names.append(row['source_path']);pins[str(path)]=sha(path)
assert sha(P/'AUTHOR_TREE.txt')==m['tree_sha256']
actual=subprocess.check_output(['git','ls-tree','-r','-l',m['author_revision'],'--','m16/approaches/neuron_account/'],cwd=R)
assert actual==(P/'AUTHOR_TREE.txt').read_bytes()
for path,digest in m['prior_pins'].items():assert sha(path)==digest;pins[path]=digest
for n in ['manifest.json','SOURCE_REVIEW.md','README.md','AUTHOR_TREE.txt','provenance_result.json']:pins[str(P/n)]=sha(P/n)
for path in [Path(__file__),HERE/'REVIEW.md']:pins[str(path)]=sha(path)
prior=json.loads((P/'provenance_result.json').read_text());assert prior['text_provenance_pass'] and prior['author_calls']==prior['binary_payload_reads']==0
r=dict(task='T251',complete=True,pass_all=True,execution_authorized=False,execution_allowed=False,scope='Independent written algebra/source review and five-text-file metadata verification only',author_revision=m['author_revision'],reviewed_package='53da2f0c',text_files=names,pins=pins,author_calls=0,binary_result_log_payload_reads=0,algebra_review=['oblique projected norm versus removed SSE','exact two-dimensional counterexample','suffix and donor semantics','paired uncertainty incompleteness','partial recurrent sample-step formula'],empirical_scores_validated=False)
assert not (HERE/'source_review.json').exists()
(HERE/'source_review.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:r[k] for k in ['task','complete','pass_all','execution_authorized','author_calls']}))
