"""Declared inference dependency closure and framed AST constructors."""
import ast, math, struct
from pathlib import Path
P=Path(__file__).resolve().parent

def dependencies(families):
    out=set()
    for f in families:
        if type(f)is not int or not 0<=f<12:raise ValueError('family')
        out.update(range(10-{8:1,9:3,10:9,11:9}.get(f,0),10))
    return sorted(out)

def program(kind,normalize):
    if kind=='tables':
        tree=ast.parse((P/'frozen/features.py').read_text())
        tree.body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.FunctionDef) and n.name in ('_array','features')]
        tree.body+=ast.parse((P/'inference.py').read_text()).body
    elif kind=='network':tree=ast.parse((P/'network.py').read_text())
    elif kind=='HA1':
        tree=ast.parse((P/'frozen/labels.py').read_text())
        tree.body=[n for n in tree.body if not(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LABELS' for t in n.targets))]
        tree.body+=ast.parse((P/'ha_wrapper.py').read_text()).body
    else:raise ValueError('program kind')
    return normalize().visit(tree)

def measure(x):
    """Independent length recurrence; also retains scalar type counts."""
    c={'none':0,'bool':0,'int':0,'str':0,'float':0,'list':0,'tuple':0,'ast':0}
    def gamma(n):return 2*n.bit_length()-1
    def rec(z):
        if z is None:c['none']+=1;return 3
        if type(z)is bool:c['bool']+=1;return 4
        if type(z)is int:c['int']+=1;return 3+gamma((2*z if z>=0 else -2*z-1)+1)
        if type(z)is str:
            c['str']+=1;n=len(z.encode());return 3+gamma(n+1)+8*n
        if type(z)is float:
            assert math.isfinite(z);c['float']+=1;return 67
        if isinstance(z,ast.AST):c['ast']+=1;return 11+sum(rec(getattr(z,f,None)) for f in z._fields)
        if type(z)in(list,tuple):c['list' if type(z)is list else 'tuple']+=1;return 3+gamma(len(z)+1)+sum(map(rec,z))
        raise TypeError(type(z))
    return {'bits':rec(x),'counts':c}

def same(a,b):
    if type(a)is not type(b):return False
    if isinstance(a,ast.AST):return a._fields==b._fields and all(same(getattr(a,f,None),getattr(b,f,None)) for f in a._fields)
    if type(a)is float:return struct.pack('>d',a)==struct.pack('>d',b)
    if type(a)in(list,tuple):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b
