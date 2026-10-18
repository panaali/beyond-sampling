"""Load only two pinned source definitions, without m43 or stored models."""
import ast,hashlib,itertools,json,math,types
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
class Ops:
    def __init__(self):self.n=0
    def add(self,n):self.n+=n

def load():
    p=json.loads((HERE/'protocol.json').read_text());source=(HERE/'estT_snapshot.py').read_bytes()
    assert hashlib.sha256(source).hexdigest()==p['source_sha256']
    tree=ast.parse(source);wanted={'set_rules5','_orth5'}
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in body}==wanted
    m=types.ModuleType('pinned_wedge5_mass');m.__dict__.update(np=np,math=math,itertools=itertools,THIN_DET5=[1e-6],_DU5=[None],_LOG5=[None],_P5=list(itertools.combinations(range(5),2)),STATS={'simplices':0})
    exec(compile(ast.Module(body=body,type_ignores=[]),str(HERE/'estT_snapshot.py')+' [two unchanged definitions]','exec'),m.__dict__)
    m.extracted_ast_sha256=hashlib.sha256(ast.dump(ast.Module(body=body,type_ignores=[]),include_attributes=False).encode()).hexdigest()
    return m
