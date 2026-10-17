"""Transparent expression/call census; opaque kernels are not scalar priced."""
import ast
from collections import Counter
import copy
import dis
import hashlib
import operator
import numpy as np

BINARY={k:getattr(operator,v) for k,v in {'Add':'add','Sub':'sub','Mult':'mul','Div':'truediv','Pow':'pow','MatMult':'matmul'}.items()}
UNARY={'USub':operator.neg,'UAdd':operator.pos}
COMPARE={k:getattr(operator,v) for k,v in {'Lt':'lt','LtE':'le','Gt':'gt','GtE':'ge','Eq':'eq','NotEq':'ne'}.items()}
CALLS={'np.linalg.det','np.linalg.norm','np.sum','np.sqrt','np.arctan2','np.maximum','np.minimum','math.atan2','sum','min','abs'}

def compiled_constant(node):
    """Do not prevent the running CPython compiler's constant folding.

    Compile only; never evaluate a probe. A variable/call or unfurled numeric
    expression necessarily has more than a single numeric constant return.
    """
    expr=ast.Expression(body=copy.deepcopy(node));ast.fix_missing_locations(expr)
    code=compile(expr,'constant_fold_probe','eval')
    ops=[x for x in dis.get_instructions(code) if x.opname not in {'RESUME','CACHE','NOP'}]
    if len(ops)==1 and ops[0].opname=='RETURN_CONST':return isinstance(ops[0].argval,(int,float,bool))
    return len(ops)==2 and ops[0].opname=='LOAD_CONST' and isinstance(ops[0].argval,(int,float,bool)) and ops[1].opname=='RETURN_VALUE'

def fullname(node):
    if isinstance(node,ast.Name):return node.id
    if isinstance(node,ast.Attribute):return fullname(node.value)+'.'+node.attr
    return ''

def integer(x):return isinstance(x,(int,bool,np.integer,np.bool_))
def size(x):return int(x.size) if isinstance(x,np.ndarray) else 1
def metadata(x):
    if isinstance(x,np.ndarray):
        return {'kind':'array','shape':list(x.shape),'strides':list(x.strides),
                'dtype':str(x.dtype),'C':bool(x.flags.c_contiguous),'F':bool(x.flags.f_contiguous),
                'sha256':hashlib.sha256(x.tobytes(order='C')).hexdigest()}
    if isinstance(x,np.generic):return {'kind':'numpy_scalar','dtype':str(x.dtype)}
    if isinstance(x,(float,int,bool)):return {'kind':type(x).__name__}
    return {'kind':type(x).__name__,'length':len(x) if hasattr(x,'__len__') else None}

class Trace:
    def __init__(self):self.counts=Counter();self.sites=Counter();self.kernels=[]
    def record(self,site,category,op,result,args):
        count=size(result);self.counts[category+':'+op]+=count
        self.sites[(site,category,op,'calls')]+=1
        self.sites[(site,category,op,'output_elements')]+=count
        if category.startswith('opaque'):
            self.kernels.append({'site':site,'category':category,'op':op,
                                 'inputs':[metadata(a) for a in args],'output':metadata(result)})
    def binary(self,site,op,a,b):
        value=BINARY[op](a,b)
        category='integer_expression' if integer(a) and integer(b) else 'numeric_expression'
        if op=='MatMult':category='opaque_matmul'
        elif op=='Pow' and category!='integer_expression':category='opaque_power'
        self.record(site,category,op,value,(a,b));return value
    def unary(self,site,op,a):
        value=UNARY[op](a)
        self.record(site,'integer_expression' if integer(a) else 'numeric_expression',op,value,(a,));return value
    def compare(self,site,op,a,b):
        value=COMPARE[op](a,b)
        self.record(site,'integer_comparison' if integer(a) and integer(b) else 'numeric_comparison',op,value,(a,b));return value
    def call(self,site,name,fn,*args,**kwargs):
        value=fn(*args,**kwargs)
        self.record(site,'opaque_call',name,value,args);return value
    def serial(self):
        return {'category_output_elements':dict(self.counts),'sites':[
            {'site':k[0],'category':k[1],'op':k[2],'unit':k[3],'count':v}
            for k,v in sorted(self.sites.items())],'kernel_events':self.kernels,
            'scope':'Output-element expression census, opaque calls and integer categories only. No kernel-body, scalar-equivalent, memory/control/setup or full-native price.'}

class Adapt(ast.NodeTransformer):
    def wrap(self,node,method,tag,args):
        site=f'{node.lineno}:{node.col_offset}'
        return ast.copy_location(ast.Call(func=ast.Attribute(value=ast.Name(id='__trace',ctx=ast.Load()),attr=method,ctx=ast.Load()),
            args=[ast.Constant(site),ast.Constant(tag)]+args,keywords=[]),node)
    def visit_BinOp(self,node):
        if compiled_constant(node):return node
        node=self.generic_visit(node);tag=type(node.op).__name__
        assert tag in BINARY,tag
        return self.wrap(node,'binary',tag,[node.left,node.right])
    def visit_UnaryOp(self,node):
        if compiled_constant(node):return node
        node=self.generic_visit(node);tag=type(node.op).__name__
        assert tag in UNARY,tag
        return self.wrap(node,'unary',tag,[node.operand])
    def visit_Compare(self,node):
        node=self.generic_visit(node)
        assert len(node.ops)==1,'Chained comparison needs explicit short-circuit preservation'
        return self.wrap(node,'compare',type(node.ops[0]).__name__,[node.left,node.comparators[0]])
    def visit_AugAssign(self,node):
        if isinstance(node.target,ast.Subscript):
            assert ast.unparse(node.target)=="STATS['simplices']"
            return node
        assert isinstance(node.target,ast.Name)
        value=self.visit(node.value)
        # _orth5 only augments the scalar accumulator. Preserve its += by
        # using in-place operator semantics in a dedicated helper below.
        expr=self.wrap(node,'augment',type(node.op).__name__,[ast.Name(id=node.target.id,ctx=ast.Load()),value])
        return ast.copy_location(ast.Assign(targets=[node.target],value=expr),node)
    def visit_Call(self,node):
        name=fullname(node.func)
        if name=='ops.add':return node
        node=self.generic_visit(node)
        if name not in CALLS:return node
        out=self.wrap(node,'call',name,[node.func]+node.args);out.keywords=node.keywords;return out

def augment(self,site,op,a,b):
    assert op=='Add'
    value=operator.iadd(a,b)
    self.record(site,'integer_expression' if integer(a) and integer(b) else 'numeric_expression','Add',value,(a,b))
    return value
Trace.augment=augment

class Restore(ast.NodeTransformer):
    def visit_Assign(self,node):
        value=node.value
        if isinstance(value,ast.Call) and fullname(value.func)=='__trace.augment':
            assert len(node.targets)==1
            return ast.AugAssign(target=node.targets[0],op=getattr(ast,value.args[1].value)(),value=self.visit(value.args[3]))
        return self.generic_visit(node)
    def visit_Call(self,node):
        name=fullname(node.func)
        if not name.startswith('__trace.'):return self.generic_visit(node)
        method=name.split('.')[-1];tag=node.args[1].value;args=[self.visit(x) for x in node.args[2:]]
        if method=='binary':return ast.BinOp(left=args[0],op=getattr(ast,tag)(),right=args[1])
        if method=='unary':return ast.UnaryOp(op=getattr(ast,tag)(),operand=args[0])
        if method=='compare':return ast.Compare(left=args[0],ops=[getattr(ast,tag)()],comparators=[args[1]])
        assert method=='call'
        return ast.Call(func=args[0],args=args[1:],keywords=[self.visit(k) for k in node.keywords])

def transform(tree):
    original=ast.dump(tree,include_attributes=False)
    adapted=Adapt().visit(copy.deepcopy(tree));ast.fix_missing_locations(adapted)
    recovered=Restore().visit(copy.deepcopy(adapted))
    assert ast.dump(recovered,include_attributes=False)==original,'Adapter structural recovery failure'
    return adapted
