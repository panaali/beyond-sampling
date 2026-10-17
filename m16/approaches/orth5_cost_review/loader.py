"""Only pinned source definitions; no imports of models or geometry."""
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
import types
import numpy as np
from trace import Trace,transform

HERE=Path(__file__).resolve().parent
class Ops:
    def __init__(self):self.n=0
    def add(self,n):self.n+=n

def load(traced=False):
    pins=json.loads((HERE/'pins.json').read_text())
    for name,sha in pins.items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha,name
    tree=ast.parse((HERE/'estT_snapshot.py').read_text())
    wanted={'set_rules5','_orth5'}
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in body}==wanted and len(body)==2
    m=types.ModuleType('orth5_cost_fixed');m.__dict__.update(np=np,math=math,itertools=itertools,THIN_DET5=[1e-6],_DU5=[None],_LOG5=[None],_P5=list(itertools.combinations(range(5),2)),STATS={'simplices':0})
    tr=Trace();m.__dict__['__trace']=tr
    if traced:body=[transform(n) if n.name=='_orth5' else n for n in body]
    module=ast.Module(body=body,type_ignores=[])
    exec(compile(module,'pinned_orth5_two_definitions','exec'),m.__dict__)
    # Rule generation remains explicitly untraced setup and is timed by the runner.
    return m,tr
