"""Execute unchanged selected source ASTs, avoiding unrelated mass-rule setup."""
from pathlib import Path
import ast,hashlib,json
import numpy as np
HERE=Path(__file__).parent
NAMES={'Cone','_adjacent_pairs','clip','_ref','_weights_to_margin','_frozen_margin','_margin_split'}

class Ops:
    def __init__(self):self.n=0
    def add(self,n):self.n+=int(n)

def load():
    source=(HERE/'estT_frozen.py').read_bytes();m=json.loads((HERE/'manifest.json').read_text())['estT_frozen.py'];assert hashlib.sha256(source).hexdigest()==m['sha256']
    tree=ast.parse(source);nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in NAMES];assert {n.name for n in nodes}==NAMES
    env=dict(np=np,TOL=1e-10,TIE=1e-9,STATS=dict(cells=0,cut=0,resolved=0,simplices=0))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(HERE/'estT_frozen.py'),'exec'),env)
    return env
