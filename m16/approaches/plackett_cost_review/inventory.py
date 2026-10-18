"""Native-expression counting with explicitly opaque LAPACK and memory calls."""
import ast,hashlib,importlib.util,json,math,operator,itertools,types
from collections import Counter
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
AUTHOR=Path('/workspace/work/recover-small_h2_910/m16/approaches')
SOURCE=AUTHOR/'leaf_mass_review/estT_snapshot.py'
BASE=Path('/workspace/work/recover-small_h2_78/m16/approaches/fibre_cost_review/arithmetic.py')
PINS={SOURCE:'afa94627ff5b860a4b922dc340e90d0d6718c770d74e326dfbf667a2bb0bbc3a',BASE:'48bf04ead5510fc301a8db9732d90e00179d0ae6524f515344f28df2a94385db',
 AUTHOR/'leaf_mass_review/error_captured.npz':'15f9e5e483c69325c10fcbe4eaaa80f91503b177070ccec1ff2569a11c9436b6',
 AUTHOR/'leaf_mass_review/error_capture.json':'7cdd206f29c35dd5b46319b94d5ed5f5609e99873e9f0a6c73efe68fe9f48057',
 AUTHOR/'plackett_rule_doubling/run.py':'38213c96b31538b0aad2532a9c169fad60db3725d8345a39f7782d33e6531d18',
 AUTHOR/'plackett_rule_doubling/summary.json':'8b33bef924d64fa2e5cd366fb9328d9c23e76029f4e85d8fc7285b02ba934bed',
 Path('/workspace/work/t45_weights/2nd_argmax_8_4_1073741824_0.pth'):'b771f40fcae826e010b2e89fb698e2ec35026119bc9f66d3ff046dba537b07da'}
for p,sha in PINS.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,str(p)
spec=importlib.util.spec_from_file_location('reviewed_t121_base',BASE);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.BINARY['FloorDiv']=operator.floordiv

def shape_of(value):
    try:return tuple(np.shape(value))
    except ValueError:
        if isinstance(value,(tuple,list)):return ('heterogeneous',tuple(shape_of(v) for v in value))
        raise

class Ledger(base.Ledger):
    def __init__(self):super().__init__();self.opaque=Counter();self.memory=Counter();self.calls=Counter()
    def call(self,site,name,fn,*args,**kwargs):
        if name in ('min','max') and len(args)==1:
            seen=0;category='integer_control'
            def values():
                nonlocal seen,category
                for value in args[0]:
                    seen+=1
                    if self.category(value)!='integer_control':category='scalar'
                    yield value
            result=fn(values(),**kwargs);self.charge(site,category+'_compare',max(seen-1,0));return result
        if name=='sum':
            def values():
                for value in args[0]:
                    category=self.category(value) or 'integer_control';self.charge(site,category+'_sum');yield value
            return fn(values(),*args[1:],**kwargs)
        result=fn(*args,**kwargs)
        receiver=fn.__self__ if name.startswith('.') else (args[0] if args else None)
        shape=shape_of(receiver);self.calls[(name,str(shape),str(np.shape(result)))]+=1
        if name in ('np.linalg.inv','np.linalg.det','np.linalg.matrix_rank','np.polynomial.legendre.leggauss'):
            self.opaque[(site,name,str(shape),str(np.shape(result)),str(kwargs))]+=1
        elif name=='np.linalg.norm':
            assert kwargs.get('ord') is None and (len(args)==1 or args[1] is None)
            n=np.size(receiver);m=np.size(result)
            self.charge(site,'norm_mul',n);self.charge(site,'norm_add',max(n-m,0));self.charge(site,'special_sqrt',m)
        elif name in ('np.abs','np.sqrt','np.isnan') or name.startswith('math.') or name=='abs':
            self.charge(site,'special_'+name,np.size(result))
        elif name=='np.outer':self.charge(site,'array_Mult',np.size(result))
        elif name in ('np.minimum','np.maximum'):self.charge(site,'array_compare',np.size(result))
        elif name=='np.clip':self.charge(site,'array_compare',2*np.size(result))
        elif name in ('np.min','np.max','np.sum','np.any','np.all','np.argmax','np.argmin','.min','.max','.sum','.mean','.any','.all'):
            n=np.size(receiver);m=np.size(result);self.charge(site,'reduce_'+name,max(n-m,0))
            if name=='.mean':self.charge(site,'array_Div',m)
        elif name in ('min','max'):
            if len(args)==1:raise ValueError('single iterable min/max requires separate traced iterator')
            cat='integer_control' if all(isinstance(v,(int,np.integer)) for v in args) else 'scalar'
            self.charge(site,cat+'_compare',len(args)-1)
        elif name in ('all','any'):
            # Builtin short-circuit reductions are instrumented by Rewrite's iterator wrapper.
            raise AssertionError('builtin reduction must use stream path')
        elif name in ('np.array','np.asarray','np.zeros','np.eye','np.diag','np.vstack','np.hstack','np.append','np.flatnonzero','np.delete','.copy','.astype'):
            self.memory[(site,name,str(shape),str(np.shape(result)))]+=1
        else:raise ValueError('unclassified native call '+name)
        return result
    def stream(self,site,name,fn,iterable):
        seen=0
        def values():
            nonlocal seen
            for value in iterable:
                if seen:self.charge(site,'integer_control_reduce_'+name)
                seen+=1;yield value
        return fn(values())
    def report(self):
        out=super().report();out.update(inclusive_explicit_total=out['named_numeric_total']+out['integer_control_total'],opaque_calls=[dict(key=list(k),count=v) for k,v in sorted(self.opaque.items())],memory_calls=[dict(key=list(k),count=v) for k,v in sorted(self.memory.items())],native_calls=[dict(key=list(k),count=v) for k,v in sorted(self.calls.items())]);return out

class Rewrite(base.Rewrite):
    def visit_AugAssign(self,node):
        if isinstance(node.target,ast.Subscript) and isinstance(node.target.value,ast.Name) and node.target.value.id=='STATS':return node
        return super().visit_AugAssign(node)
    def visit_Call(self,node):
        name=ast.unparse(node.func)
        if isinstance(node.func,ast.Attribute) and node.func.attr in ('min','max','sum','mean','any','all','copy','astype') and not name.startswith(('np.','math.')):name='.'+node.func.attr
        if name in ('all','any'):
            site=self.site(node);node=self.generic_visit(node);assert len(node.args)==1 and not node.keywords
            return self.wrap(node,'stream',[ast.Constant(site),ast.Constant(name),node.func,node.args[0]])
        if name.startswith(('np.','math.')) or name in ('sum','abs','min','max','.min','.max','.sum','.mean','.any','.all','.copy','.astype'):
            site=self.site(node);node=self.generic_visit(node);out=self.wrap(node,'call',[ast.Constant(site),ast.Constant(name),node.func]+node.args);out.keywords=node.keywords;return out
        return super().visit_Call(node)

class Ops:
    def __init__(self):self.n=0
    def add(self,n):self.n+=n


def load(instrument=True):
    module=types.ModuleType('pinned_estT_cost');ledger=Ledger()
    module.__dict__.update(np=np,math=math,itertools=itertools,Ops=Ops,_arithmetic=ledger)
    tree=ast.parse(SOURCE.read_text());tree.body=[node for node in tree.body if isinstance(node,(ast.Assign,ast.FunctionDef,ast.ClassDef))]
    # Module constants/rule setup are instrumented but reported separately.
    if instrument:tree=Rewrite().visit(tree)
    exec(compile(ast.fix_missing_locations(tree),str(SOURCE)+' [native trace]' if instrument else str(SOURCE), 'exec'),module.__dict__)
    module._RULES={'wide':module._rule(24,2),'thin':module._rule(96,3)}
    startup=ledger.report();ledger=Ledger();module._arithmetic=ledger
    return module,ledger,startup
