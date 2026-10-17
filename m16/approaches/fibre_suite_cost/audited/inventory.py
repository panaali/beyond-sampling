"""T121 reviewed native wrapper, extended only for pinned fast-path primitives."""
import ast,operator
import numpy as np
import arithmetic_base as base
base.BINARY['FloorDiv']=operator.floordiv

class Ledger(base.Ledger):
    def call(self,site,name,fn,*args,**kwargs):
        if name=='sum':
            category='integer_control'
            def values():
                nonlocal category
                for value in args[0]:
                    if self.category(value)!='integer_control':category='scalar'
                    self.charge(site,category+'_sum');yield value
            return fn(values(),*args[1:],**kwargs)
        if name in ('np.outer','np.minimum','np.maximum','np.hypot','np.array_equal','divmod'):
            result=fn(*args,**kwargs)
            if name=='np.outer':self.charge(site,'array_Mult',np.size(result))
            elif name in ('np.minimum','np.maximum'):self.charge(site,'array_compare',np.size(result))
            elif name=='np.hypot':self.charge(site,'special_np.hypot',np.size(result))
            elif name=='np.array_equal':
                assert np.shape(args[0])==np.shape(args[1]),'unreviewed shape mismatch equality branch'
                n=np.size(args[0]);self.charge(site,'array_compare',n);self.charge(site,'reduce_array_equal',max(n-1,0))
            else:self.charge(site,'integer_control_divmod',2)
            return result
        if name in ('min','max') and all(isinstance(x,(int,np.integer)) for x in args):
            result=fn(*args,**kwargs);self.charge(site,'integer_control_compare',len(args)-1);return result
        return super().call(site,name,fn,*args,**kwargs)

class Rewrite(base.Rewrite):
    def __init__(self,source='primitive'):super().__init__();self.source=source
    def site(self,node):return self.source+':'+super().site(node)
    def visit_Call(self,node):
        name=ast.unparse(node.func)
        if name in ('np.outer','np.minimum','np.maximum','np.hypot','np.array_equal','divmod','sum'):
            site=self.site(node);node=self.generic_visit(node);out=self.wrap(node,'call',[ast.Constant(site),ast.Constant(name),node.func]+node.args);out.keywords=node.keywords;return out
        return super().visit_Call(node)
