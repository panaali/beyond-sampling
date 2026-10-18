"""Observe the unchanged Q expression tree; indexing/native internals excluded."""
import ast,operator
from collections import Counter
import numpy as np
from scipy.special import ndtr,stdtr
LEDGER_NAME='_srb_observer_ledger'
OPS={'Add':operator.add,'Sub':operator.sub,'Mult':operator.mul,'Div':operator.truediv,'MatMult':operator.matmul,'BitAnd':operator.and_,'BitOr':operator.or_,'FloorDiv':operator.floordiv}
UNARY={'USub':operator.neg,'UAdd':operator.pos,'Invert':operator.invert,'Not':operator.not_}
CMP={'Gt':operator.gt,'Lt':operator.lt,'GtE':operator.ge,'LtE':operator.le,'Eq':operator.eq,'NotEq':operator.ne}
def meta(x):
    if isinstance(x,np.ndarray):return dict(shape=list(x.shape),dtype=str(x.dtype))
    if isinstance(x,np.generic):return dict(shape=[],dtype=str(x.dtype))
    if isinstance(x,(int,float,bool)):return dict(shape=[],dtype=type(x).__name__)
    if isinstance(x,(list,tuple)):return [meta(v) for v in x]
    return dict(type=type(x).__name__)
def size(x):return int(np.size(x)) if isinstance(x,(np.ndarray,np.generic)) else 1
class Ledger:
    def __init__(self):self.events=[];self.intervals=[];self.counts=Counter()
    def result(self,site,kind,inputs,out):
        n=size(out);dt=getattr(out,'dtype',None)
        category='float' if (dt is not None and dt.kind in 'fc') or type(out) is float else 'bool' if (dt is not None and dt.kind=='b') or type(out) is bool else 'integer_control'
        self.events.append(dict(site=site,kind=kind,inputs=[meta(x) for x in inputs],output=meta(out),units=n,category=category))
        if kind!='MatMult':self.counts[category+('_array' if isinstance(out,np.ndarray) else '_scalar')]+=n
        else:
            a,b=inputs;assert a.ndim==b.ndim==2 and a.shape[1]==b.shape[0]
            m,k=a.shape;nout=b.shape[1]
            self.counts['dense_2mac']+=2*m*k*nout;self.counts['dense_dot_expression']+=m*nout*(2*k-1)
        return out
    def binary(self,site,kind,a,b):return self.result(site,kind,[a,b],OPS[kind](a,b))
    def unary(self,site,kind,a):return self.result(site,kind,[a],UNARY[kind](a))
    def compare(self,site,kind,a,b):return self.result(site,kind,[a,b],CMP[kind](a,b))
    def aug_rhs(self,site,kind,a,b):
        # The original in-place operator still executes once in its original statement.
        n=int(np.prod(np.broadcast_shapes(np.shape(a),np.shape(b))))
        assert isinstance(a,np.ndarray) and a.dtype.kind=='f'
        self.events.append(dict(site=site,kind='inplace_'+kind,inputs=[meta(a),meta(b)],output=meta(a),units=n,category='float'));self.counts['float_array']+=n
        return b
    def call(self,site,fn,*args,**kwargs):
        out=fn(*args,**kwargs)
        bound=getattr(fn,'__self__',None)
        self.events.append(dict(site=site,kind='call',name=getattr(fn,'__name__',type(fn).__name__),bound=meta(bound) if isinstance(bound,np.ndarray) else None,inputs=[meta(a) for a in args],kwargs={k:meta(v) for k,v in kwargs.items()},output=meta(out)))
        return out
    def interval(self,k,ok):self.intervals.append(dict(k=int(k),mask=ok.tolist(),active=bool(ok.any())))
    def report(self):return dict(counts=dict(self.counts),intervals=self.intervals,events=self.events)
class Observe(ast.NodeTransformer):
    def __init__(self):self.function=''
    def visit_FunctionDef(self,node):
        old=self.function;self.function=node.name;node=self.generic_visit(node);self.function=old;return node
    def site(self,node):return f'{self.function}:{node.lineno}:{node.col_offset}'
    def call(self,node,method,args):
        return ast.copy_location(ast.Call(ast.Attribute(ast.Name(LEDGER_NAME,ast.Load()),method,ast.Load()),[ast.Constant(self.site(node)),*args],[]),node)
    def visit_BinOp(self,node):
        kind=type(node.op).__name__;assert kind in OPS,kind
        return self.call(node,'binary',[ast.Constant(kind),self.visit(node.left),self.visit(node.right)])
    def visit_UnaryOp(self,node):
        kind=type(node.op).__name__;assert kind in UNARY,kind
        return self.call(node,'unary',[ast.Constant(kind),self.visit(node.operand)])
    def visit_Compare(self,node):
        assert len(node.ops)==len(node.comparators)==1
        kind=type(node.ops[0]).__name__;assert kind in CMP,kind
        return self.call(node,'compare',[ast.Constant(kind),self.visit(node.left),self.visit(node.comparators[0])])
    def visit_Call(self,node):
        f=self.visit(node.func);args=[self.visit(x) for x in node.args];kw=[ast.keyword(k.arg,self.visit(k.value)) for k in node.keywords]
        out=self.call(node,'call',[f,*args]);out.keywords=kw;return out
    def visit_AugAssign(self,node):
        assert isinstance(node.target,ast.Name)
        rhs=self.call(node,'aug_rhs',[ast.Constant(type(node.op).__name__),ast.Name(node.target.id,ast.Load()),self.visit(node.value)])
        return ast.copy_location(ast.AugAssign(node.target,node.op,rhs),node)
    def visit_Assign(self,node):
        out=self.generic_visit(node)
        if self.function=='Q' and any(isinstance(t,ast.Name) and t.id=='ok' for t in node.targets):
            extra=ast.Expr(ast.Call(ast.Attribute(ast.Name(LEDGER_NAME,ast.Load()),'interval',ast.Load()),[ast.Name('k',ast.Load()),ast.Name('ok',ast.Load())],[]))
            return [out,ast.copy_location(extra,node)]
        return out

def load(source,traced=False):
    parsed=ast.parse(source);nodes=[n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name in ('h9','Q')]
    assert [n.name for n in nodes]==['h9','Q']
    tree=ast.Module(body=nodes,type_ignores=[])
    bindings={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)} | {n.arg for n in ast.walk(tree) if isinstance(n,ast.arg)} | {n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
    assert LEDGER_NAME not in bindings, 'observer name collides with original source'
    if traced:tree=Observe().visit(tree)
    ast.fix_missing_locations(tree)
    env=dict(np=np,ndtr=ndtr,stdtr=stdtr,T=10)
    exec(compile(tree,'<pinned-rb9-trace>' if traced else '<pinned-rb9-original>','exec'),env)
    return env,ast.unparse(tree)
