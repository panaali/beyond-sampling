"""Pinned source/data validation and exact existing-panel reconstruction."""
import ast,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).parent
DATA=Path('/workspace/clean-neurons/m16/approaches/fibre_constraint_review')
sys.path.insert(0,str(HERE/'audited'))
from loader import load
from sampler_native import count as sample_count

def verify_sources():
    rows=json.loads((HERE/'source_manifest.json').read_text())['files']
    for row in rows:
        path=HERE/row['local'] if 'local' in row else Path(row['source']);assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],str(path)
    class StripX(ast.NodeTransformer):
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='X':return None
            return self.generic_visit(node)
    cur=ast.parse((HERE/'audited/snapshot/fibre.py').read_text());old=ast.parse((HERE/'audited/snapshot/fibre_authorcount.py').read_text());checks={}
    for name in ('rows_at','State','_roots_vec','Cons','trace_arc_rows'):
        a=next(n for n in cur.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name);b=next(n for n in old.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name);checks[name]=ast.dump(StripX().visit(a),include_attributes=False)==ast.dump(b,include_attributes=False)
    assert all(checks.values());return checks

def models(h):
    # Execute only the exact frozen reconstruction function, never its old truth runner.
    tree=ast.parse((DATA/'suite.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='models');scope=dict(HERE=DATA,json=json,Path=Path,hashlib=hashlib,torch=torch,np=np);exec(compile(ast.Module(body=[fn],type_ignores=[]),str(DATA/'suite.py')+' [models only]','exec'),scope);return scope['models'](h)

def references(h):
    original=json.loads((DATA/'suite_reference.json').read_text())[str(h)]['rows'];prior={}
    for path in sorted(DATA.glob(f'suite_h{h}_*.json')):
        chunk=json.loads(path.read_text());assert chunk['complete']
        for row in chunk['models']:
            assert row['name'] not in prior;prior[row['name']]=row
    assert len(prior)==157 and list(prior)==list(original);return original,prior

def digest(model):return hashlib.sha256(b''.join(w.tobytes() for w in model)).hexdigest()

def evaluate(model):
    mod,mods,ledger=load(True);assert mod.LOOK==2 and all(getattr(mod,k) for k in ('PRUNE','RANGEF','INCR','CLAMPC','FLIP','SHARE'));helper=[0];oldX=mods['fibre'].X
    def charge(ops,k):helper[0]+=k;oldX(ops,k)
    mods['fibre'].X=charge;ops=mod.Ops();info={};value=mod.est_fast(model,ops,info);report=ledger.report()
    return dict(value=float(value),value_hex=float(value).hex(),current_n=ops.n,current_helper_X=helper[0],original_n=ops.n-helper[0],segments=info['segments'],traces=info['traces'],cons=info['cons'],family_hex={str(k):float(v).hex() for k,v in info['err_by'].items()},ledger=report,expression_total=report['named_numeric_total']+report['integer_control_total'])
