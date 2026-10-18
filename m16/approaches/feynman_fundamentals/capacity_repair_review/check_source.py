"""Source-only assertions; no author imports or numerical calls."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
P=Path(__file__).resolve().parent
R=P.parents[3]
m=json.loads((P/'manifest.json').read_text())
assert len(m['files'])==11
changed=[]
for f in m['files']:
    b=(R/f['local']).read_bytes()
    assert b==subprocess.check_output(['git','show',f['source_commit']+':'+f['source_path']],cwd=R)
    assert len(b)==f['bytes'] and hashlib.sha256(b).hexdigest()==f['sha256']
    assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==f['git_blob']
    old=subprocess.check_output(['git','show',m['prior_source']+':'+f['source_path']],cwd=R)
    assert (b!=old)==f['changed_from_prior']
    if b!=old: changed.append(Path(f['local']).name)
    if f['local'].endswith('.py'): ast.parse(b)
assert sorted(changed)==['README.md','evaluate.py','run.py','trainer.py']
def source(n): return (P/'snapshot'/n).read_text()
def tree(n): return ast.parse(source(n))
def fn(n,name): return next(x for x in ast.walk(tree(n)) if isinstance(x,ast.FunctionDef) and x.name==name)
def norm(n): return ast.unparse(n)
e=fn('evaluate.py','evaluate')
t=next(x for x in ast.walk(e) if isinstance(x,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='third' for v in x.targets))
assert isinstance(t.value,ast.IfExp) and norm(t.value.test)=='T >= 3'
assert norm(t.value.body)=='order[:, -3]' and norm(t.value.orelse)=='th.full_like(lab, -1)'
r=norm(tree('run.py'))
for fragment in ["if st.get('meta') != meta:","raise SystemExit",'if s % BLOCK == 0 or s == n_steps:', 'tr.reseed(a.data_seed, (start - 1) // BLOCK)', 'tr.reseed(a.data_seed, s // BLOCK)',"('trainer.py', 'run.py')"]: assert fragment in r,fragment
assert 'BLOCK = 2 ** 18' in r
assert 'seed = data_seed * 1000003 + block' in norm(fn('trainer.py','reseed'))
assert 'manual_seed' not in norm(fn('trainer.py','load_state'))
assert 'correct=int(c[0])' in norm(e) and 'discordant_correct_vs=disc[k].tolist()' in norm(e)
assert 'O @ (1 - O).T' in norm(e)
assert source('ht_table.py')==subprocess.check_output(['git','show',m['prior_source']+':m16/approaches/capacity_ceiling/ht_table.py'],cwd=R).decode()
print(json.dumps(dict(complete=True,pass_all=True,task='T249',source_files=11,changed_files=changed,source_corrections_verified=['T2 third rank guard','exact metadata comparison','block reseed schedule','final state save','within-call integer correct and discordance counts'],execution_authorized=False,runtime_resume_verified=False,historical_run_verified=False,author_calls=0,finite_control_evidence='not provided in frozen source delta'),indent=2))
