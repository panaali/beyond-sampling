"""Pinned unchanged arithmetic with an independent named-kernel ledger."""
import ast,hashlib,types,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
HERE=Path(__file__).parent

class Ledger:
    def __init__(self):self.rows=defaultdict(lambda:dict(calls=0,historical_mac=0,dense_mac=0,mul=0,add=0,mask_mul=0))
    def matmul(self,site,a,b,historical=None):
        a=np.asarray(a);b=np.asarray(b)
        if a.ndim not in (1,2) or b.ndim not in (1,2) or a.shape[-1]!=b.shape[0]:raise ValueError(('unsupported dimensions',a.shape,b.shape))
        k=a.shape[-1];outputs=(a.shape[0] if a.ndim==2 else 1)*(b.shape[1] if b.ndim==2 else 1)
        row=self.rows[site];row['calls']+=1;row['dense_mac']+=outputs*k;row['historical_mac']+=outputs*k if historical is None else int(historical)
        row['mul']+=outputs*k;row['add']+=outputs*max(k-1,0)
        return a@b
    def mask(self,site,a,b):
        result=a*b;row=self.rows[site];row['calls']+=1;row['mask_mul']+=result.size
        return result
    def report(self):
        totals={k:sum(r[k] for r in self.rows.values()) for k in ('calls','historical_mac','dense_mac','mul','add','mask_mul')}
        return {'sites':dict(sorted(self.rows.items())),'totals':totals,'scope':'only executed matrix products and gate-mask products; other hybrid scalar categories remain unaudited'}

class Rewrite(ast.NodeTransformer):
    def __init__(self):self.function='module';self.sites=[]
    def visit_FunctionDef(self,node):
        previous=self.function;self.function=node.name;node=self.generic_visit(node);self.function=previous;return node
    def visit_BinOp(self,node):
        expression=ast.unparse(node);site=f'{self.function}:L{node.lineno}:{expression}';node=self.generic_visit(node)
        if isinstance(node.op,ast.MatMult):
            historical=None
            if self.function=='rows_at':
                historical={'Wh @ Ag':'H * int(g.sum()) * t','A @ x':'H * (t+1)','Wo @ Ag':'T * int(g.sum()) * T'}.get(expression)
            args=[ast.Constant(site),node.left,node.right]
            if historical:args.append(ast.parse(historical,mode='eval').body)
            self.sites.append({'site':site,'historical_expression':historical or 'dense MAC dimensions'})
            return ast.copy_location(ast.Call(func=ast.Attribute(value=ast.Name('_ledger',ast.Load()),attr='matmul',ctx=ast.Load()),args=args,keywords=[]),node)
        if self.function=='rows_at' and expression=='A * g[:, None]':
            self.sites.append({'site':site,'historical_expression':'not charged in pinned source'})
            return ast.copy_location(ast.Call(func=ast.Attribute(value=ast.Name('_ledger',ast.Load()),attr='mask',ctx=ast.Load()),args=[ast.Constant(site),node.left,node.right],keywords=[]),node)
        return node

def load(instrument=False):
    dep_path=HERE/'snapshot/fiber3.py';dep=types.ModuleType('fiber3');dep.__file__=str(dep_path);exec(compile(dep_path.read_text(),str(dep_path),'exec'),dep.__dict__)
    path=HERE/'snapshot/fibre_strict.py';src=path.read_text();anchor='    stc = State(model, x, P, U, nn, ops)\n';assert src.count(anchor)==1
    src=src.replace(anchor,anchor+'    cache.setdefault("_alive", []).append(stc)\n');mod=types.ModuleType('local_fibre_review');mod.__file__=str(path);ledger=Ledger();mod._ledger=ledger
    tree=ast.parse(src,filename=str(path));rewrite=Rewrite()
    if instrument:tree=ast.fix_missing_locations(rewrite.visit(tree))
    previous=sys.modules.get('fiber3');sys.modules['fiber3']=dep
    try:exec(compile(tree,str(path)+' [local R4 ledger]','exec'),mod.__dict__)
    finally:
        if previous is None:del sys.modules['fiber3']
        else:sys.modules['fiber3']=previous
    mod.LOOK=3
    manifest={'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'dependency_sha256':hashlib.sha256(dep_path.read_bytes()).hexdigest(),'r4_source_sha256':hashlib.sha256(src.encode()).hexdigest(),'sites':rewrite.sites,'look':3}
    return mod,ledger,manifest

def evaluate(model,instrument=False):
    mod,ledger,manifest=load(instrument);ops=mod.Ops();info={};value=mod.est_fibre2(model,ops,info);report=ledger.report();tot=report['totals'];historical=ops.n+getattr(ops,'x',0)
    return {'value':value,'value_hex':float(value).hex(),'n':ops.n,'x':getattr(ops,'x',0),'historical_hybrid':historical,'segments':info.get('segments'),'traces':info.get('traces'),'ledger':report,'partial_scalar_adjustment':historical-tot['historical_mac']+tot['mul']+tot['add']+tot['mask_mul'],'complete_scalar_count':False,'manifest':manifest}
