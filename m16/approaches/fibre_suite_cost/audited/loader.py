"""Fresh isolated frozen modules; no sys.path or persistent module-registry changes."""
import types,sys,ast
from pathlib import Path
HERE=Path(__file__).parent

def load(instrument=False):
    mods={};old={name:sys.modules.get(name) for name in ('fiber3','fibre')}
    ledger=None
    if instrument:
        from inventory import Ledger,Rewrite
        ledger=Ledger()
    try:
        for name in ('fiber3','fibre','fibre_fast'):
            path=HERE/'snapshot'/f'{name}.py';mod=types.ModuleType('frozen_'+name);mod.__file__=str(path);src=path.read_text();exec(compile(src,str(path),'exec'),mod.__dict__)
            if instrument:
                mod._arithmetic=ledger;tree=ast.parse(src)
                # Keep already-created data-independent constants/imports; replace executable definitions.
                if name=='fiber3':tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_Sint']
                else:tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
                exec(compile(ast.fix_missing_locations(Rewrite(name).visit(tree)),str(path)+' [native inventory]','exec'),mod.__dict__)
            mods[name]=mod
            if name in old:sys.modules[name]=mod
    finally:
        for name,prior in old.items():
            if prior is None:sys.modules.pop(name,None)
            else:sys.modules[name]=prior
    return mods['fibre_fast'],mods,ledger
