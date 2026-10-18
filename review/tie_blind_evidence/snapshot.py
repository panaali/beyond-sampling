"""Metadata-only git snapshot; imports no author module."""
import hashlib,json,pathlib,subprocess
HERE=pathlib.Path(__file__).resolve().parent
REV='2809650d'
def git(*args): return subprocess.check_output(['git',*args])
commit=git('rev-parse',REV).decode().strip()
paths=git('ls-tree','-r','--name-only',commit,'--','m16/approaches/tie_blind/').decode().splitlines()
paths += ['m16/approaches/sci_tao/core.py','m16/approaches/sci_tao/estimator.py','m16/approaches/sci_tao/README.md','m16/leaderboard.py','m16/arc_eval.py','m16/common.py','results/leaderboard_truths.json']
assert not (HERE/'manifest.json').exists()
rows=[]
for path in paths:
    data=git('show',f'{commit}:{path}'); dest=HERE/'frozen'/path
    assert not dest.exists(); dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(data)
    rows.append(dict(original_path=path,snapshot_path=str(dest.relative_to(HERE)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),git_blob=git('rev-parse',f'{commit}:{path}').decode().strip()))
path='results/leaderboard_v3_raw.jsonl'; data=git('show',f'{commit}:{path}')
external=dict(original_path=path,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),git_blob=git('rev-parse',f'{commit}:{path}').decode().strip(),note='Large historical source raw pinned only; scoped copied sci_tao rows retained in author lb/raw.jsonl. No numerical contraction in this phase.')
(HERE/'manifest.json').write_text(json.dumps(dict(task='T226',author_commit=commit,files=rows,external_git_blob=[external],execution='Metadata only; zero author imports, models, RNG, scores, MC or geometry. External AlgZoo/NumPy/SciPy runtime and checkpoint bytes not recovered.'),indent=2)+'\n')
