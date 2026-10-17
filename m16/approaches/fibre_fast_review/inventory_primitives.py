import ast,json
from pathlib import Path
import numpy as np
from inventory import Ledger,Rewrite

def main():
    src='''
def run(a,b):
    outer=np.outer(a,b)
    lo=np.minimum(a,b);hi=np.maximum(a,b)
    hyp=np.hypot(a,b)
    equal=np.array_equal(a,b)
    q,r=divmod(7,3)
    floored=a//2
    total=sum([1.,2.,3.])
    ints=sum(k for k in [1,2,3])
    return outer,lo,hi,hyp,equal,q,r,floored,total,ints
'''
    a=np.array([1.,2.,3.]);b=np.array([2.,2.,4.]);L=Ledger();scope={'_arithmetic':L,'np':np};exec(compile(ast.fix_missing_locations(Rewrite().visit(ast.parse(src))),'fast inventory controls','exec'),scope);plain={'np':np};exec(src,plain)
    actual=scope['run'](a,b);expected=plain['run'](a,b);assert all(np.array_equal(x,y) for x,y in zip(actual,expected));r=L.report();t=r['totals']
    expected_counts={'array_Mult':9,'array_compare':9,'special_np.hypot':3,'reduce_array_equal':2,'integer_control_divmod':2,'array_FloorDiv':3,'scalar_sum':3,'integer_control_sum':3};assert t==expected_counts,(t,expected_counts)
    Path(__file__).with_name('inventory_primitive_results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(t))
if __name__=='__main__':main()
