"""Native-operation inventory; control/index overhead remains separately visible."""
import ast,copy,operator,types
from collections import defaultdict
import numpy as np

BINARY={name:getattr(operator,func) for name,func in [('Add','add'),('Sub','sub'),('Mult','mul'),('Div','truediv'),('Mod','mod'),('Pow','pow'),('BitAnd','and_'),('BitOr','or_'),('MatMult','matmul')]}
UNARY={'USub':operator.neg,'UAdd':operator.pos,'Invert':operator.invert,'Not':operator.not_}
COMPARE={'Lt':operator.lt,'LtE':operator.le,'Gt':operator.gt,'GtE':operator.ge,'Eq':operator.eq,'NotEq':operator.ne,'Is':operator.is_,'IsNot':operator.is_not,'In':lambda a,b:a in b,'NotIn':lambda a,b:a not in b}

class Ledger:
    def __init__(self):self.rows=defaultdict(lambda:defaultdict(int))
    def charge(self,site,kind,n=1):self.rows[site][kind]+=int(n)
    def category(self,a,b=None):
        values=(a,) if b is None else (a,b)
        if any(isinstance(v,np.ndarray) for v in values):return 'array'
        if any(isinstance(v,(float,np.floating,complex,np.complexfloating)) for v in values):return 'scalar'
        if all(isinstance(v,(int,np.integer,bool,np.bool_)) for v in values):return 'integer_control'
        return None
    def size(self,a,b=None):
        if b is None:return np.size(a)
        return int(np.prod(np.broadcast_shapes(np.shape(a),np.shape(b)),dtype=int))
    def binary(self,site,kind,a,b):
        result=BINARY[kind](a,b);category=self.category(a,b)
        if kind=='MatMult':
            k=np.shape(a)[-1];n=np.size(result);self.charge(site,'matrix_mul',n*k);self.charge(site,'matrix_add',n*max(k-1,0))
        elif category:self.charge(site,category+'_'+kind,self.size(result))
        return result
    def unary(self,site,kind,a):
        result=UNARY[kind](a);category=self.category(a)
        # Python `not` and scalar bool combinations are control conventions, not array arithmetic.
        if category:self.charge(site,('integer_control' if kind=='Not' else category)+'_'+kind,self.size(a))
        return result
    def chain(self,site,kinds,thunks):
        a=thunks[0]()
        for i,kind in enumerate(kinds):
            b=thunks[i+1]();result=COMPARE[kind](a,b)
            if kind not in ('Is','IsNot','In','NotIn'):
                category=self.category(a,b)
                if category:self.charge(site,category+'_compare',self.size(result))
            if i<len(kinds)-1 and not result:return result
            a=b
        return result
    def aug_rhs(self,site,kind,a,b):
        category=self.category(a,b)
        if category:self.charge(site,category+'_'+kind,self.size(a,b))
        return b
    def call(self,site,name,fn,*args,**kwargs):
        result=fn(*args,**kwargs)
        if name in ('np.abs','np.isnan','np.arctan2') or name.startswith('math.') or name=='abs':
            self.charge(site,'special_'+name,np.size(result))
        elif name in ('np.argmax','np.argmin') or name in ('.min','.sum','.any'):
            n=np.size(args[0]) if name.startswith('np.') else np.size(fn.__self__)
            self.charge(site,'reduce_'+name,max(n-1,0))
        elif name in ('min','max'):self.charge(site,'scalar_compare',len(args)-1)
        return result
    def sort(self,site,fn,*args,**kwargs):
        key=kwargs.get('key',lambda x:x);ledger=self
        class Key:
            def __init__(self,value):self.value=value
            def __lt__(self,other):ledger.charge(site,'sort_compare');return self.value<other.value
        kwargs['key']=lambda x:Key(key(x))
        return fn(*args,**kwargs)
    def report(self):
        totals=defaultdict(int)
        for row in self.rows.values():
            for key,value in row.items():totals[key]+=value
        integer=sum(v for k,v in totals.items() if k.startswith('integer_control'))
        return dict(sites={k:dict(v) for k,v in sorted(self.rows.items())},totals=dict(sorted(totals.items())),named_numeric_total=sum(totals.values())-integer,integer_control_total=integer,complete_implementation_cost=False)

class Rewrite(ast.NodeTransformer):
    def __init__(self):self.function='module'
    def site(self,node):return f'{self.function}:L{node.lineno}:{ast.unparse(node)}'
    def wrap(self,node,method,args):return ast.copy_location(ast.Call(ast.Attribute(ast.Name('_arithmetic',ast.Load()),method,ast.Load()),args,[]),node)
    def visit_FunctionDef(self,node):
        if node.name=='X':return node
        before=self.function;self.function=node.name;node=self.generic_visit(node);self.function=before;return node
    def visit_Call(self,node):
        # Historical accounting is diagnostic work, never estimator arithmetic.
        if isinstance(node.func,ast.Name) and node.func.id=='X':return node
        if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='ops':return node
        site=self.site(node);name=ast.unparse(node.func);node=self.generic_visit(node)
        if name=='sorted' or isinstance(node.func,ast.Attribute) and node.func.attr=='sort':
            out=self.wrap(node,'sort',[ast.Constant(site),node.func]+node.args);out.keywords=node.keywords;return out
        name='.'+node.func.attr if isinstance(node.func,ast.Attribute) and node.func.attr in ('min','sum','any') else name
        if name.startswith('math.') or name in ('np.abs','np.isnan','np.arctan2','np.argmax','np.argmin','.min','.sum','.any','min','max','abs'):
            out=self.wrap(node,'call',[ast.Constant(site),ast.Constant(name),node.func]+node.args);out.keywords=node.keywords;return out
        return node
    def visit_BinOp(self,node):
        if isinstance(node.left,ast.Constant) and isinstance(node.right,ast.Constant):return node
        site=self.site(node);node=self.generic_visit(node);kind=type(node.op).__name__
        assert kind in BINARY,kind
        return self.wrap(node,'binary',[ast.Constant(site),ast.Constant(kind),node.left,node.right])
    def visit_UnaryOp(self,node):
        if isinstance(node.operand,ast.Constant):return node
        site=self.site(node);node=self.generic_visit(node)
        return self.wrap(node,'unary',[ast.Constant(site),ast.Constant(type(node.op).__name__),node.operand])
    def visit_Compare(self,node):
        site=self.site(node);node=self.generic_visit(node)
        args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[])
        thunks=[ast.Lambda(copy.deepcopy(args),x) for x in [node.left]+node.comparators]
        return self.wrap(node,'chain',[ast.Constant(site),ast.List([ast.Constant(type(op).__name__) for op in node.ops],ast.Load()),ast.List(thunks,ast.Load())])
    def visit_AugAssign(self,node):
        site=self.site(node);node=self.generic_visit(node);read=copy.deepcopy(node.target);read.ctx=ast.Load()
        node.value=self.wrap(node,'aug_rhs',[ast.Constant(site),ast.Constant(type(node.op).__name__),read,node.value]);return node

def load():
    from instrument import load as baseline
    mod,_,manifest=baseline(False);ledger=Ledger();mod._arithmetic=ledger
    path=__import__('pathlib').Path(__file__).parent/'snapshot/fibre_strict.py';src=path.read_text();anchor='    stc = State(model, x, P, U, nn, ops)\n';src=src.replace(anchor,anchor+'    cache.setdefault("_alive", []).append(stc)\n')
    tree=ast.parse(src)
    # Keep imported FAMS/Ops/_Sint from the isolated pinned loader; do not redo initialization.
    tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    exec(compile(ast.fix_missing_locations(Rewrite().visit(tree)),str(path)+' [arithmetic]', 'exec'),mod.__dict__)
    dep=path.with_name('fiber3.py');tree=ast.parse(dep.read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_Sint'];assert len(tree.body)==1
    exec(compile(ast.fix_missing_locations(Rewrite().visit(tree)),str(dep)+' [_Sint arithmetic]','exec'),mod.__dict__)
    return mod,ledger,manifest

def evaluate(model):
    mod,ledger,manifest=load();ops=mod.Ops();info={};value=mod.est_fibre2(model,ops,info)
    return dict(value=value,value_hex=float(value).hex(),n=ops.n,x=ops.x,segments=info['segments'],traces=info['traces'],ledger=ledger.report(),manifest=manifest)
