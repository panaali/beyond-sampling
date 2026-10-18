"""Read-only source/AST bindings; never imports or executes author code."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
m = json.loads((BASE / 'manifest.json').read_text())
assert len(m['files']) == 11
checked = []
for row in m['files']:
    data = (ROOT / row['local']).read_bytes()
    original = subprocess.check_output(['git', 'show', row['source_commit'] + ':' + row['source_path']], cwd=ROOT)
    assert data == original
    assert len(data) == row['bytes']
    assert hashlib.sha256(data).hexdigest() == row['sha256']
    assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == row['git_blob']
    if row['local'].endswith('.py'):
        ast.parse(data)
    checked.append(row['source_path'])
def source(name):
    return (BASE/'snapshot'/name).read_text()
def function(name, fn):
    return next(n for n in ast.walk(ast.parse(source(name))) if isinstance(n, ast.FunctionDef) and n.name == fn)
def norm(node):
    return ast.unparse(node)
checks = {}
eval_fn = function('evaluate.py', 'evaluate')
third = [n for n in ast.walk(eval_fn) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='third' for t in n.targets)]
assert len(third)==1 and norm(third[0].value)=='order[:, -3]'
assert 'range(2, 11)' in ast.unparse(ast.parse(source('ht_table.py')))
checks['t2_nonempty_model_evaluation_third_rank_out_of_bounds'] = True
assert [norm(v) for v in eval_fn.args.defaults][:2] == ['10 ** 8', '777']
assert 'g.manual_seed(seed)' in norm(eval_fn)
assert '--eval-seed' not in source('run.py')
checks['default_100M_seed777_common_model_inputs'] = True
state = norm(function('trainer.py','state'))
assert all(x in state for x in ['self.params','self.m','self.v','self.step_t'])
assert 'get_state' not in state
restore = norm(function('trainer.py','load_state'))
assert "seed = data_seed * 1000003 + int(st['step'])" in restore
assert 'manual_seed(seed)' in restore
checks['moments_restored_rng_reseeded_not_serialized'] = True
run = ast.unparse(ast.parse(source('run.py')))
assert 's % 2 ** 18 == 0 and s < n_steps' in run
assert "st['curve']" in run
checks['periodic_resume_state_not_final_state'] = True
test = ast.unparse(ast.parse(source('test_equiv.py')))
assert 'graph=False' in test and '300' in test
checks['finite_equivalence_check_graph_disabled'] = True
cap = norm(function('trainer.py','_capture'))
assert 'range(3)' in cap and 'self.step_t.zero_()' in cap and 'with th.cuda.graph(self.graph)' in cap
checks['capture_inspection_does_not_infer_device_execution_during_recording'] = True
assert 'set -e' not in source('q_ht.sh') and 'HTDONE' in source('q_ht.sh')
checks['shell_completion_marker_not_success_guard'] = True
print(json.dumps({'complete':True,'pass_all':True,'meaning':'frozen source bindings and listed structural assertions only', 'files':len(checked),'checks':checks,'author_imports':0,'model_calls':0,'execution_authorized':False},indent=2))
