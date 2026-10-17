import ast,json
from pathlib import Path
import numpy as np
from arithmetic import Ledger,Rewrite,load

def main():
    mod,L,_=load();model=(np.zeros(4),np.zeros((4,4)),np.zeros((3,4)));ops=mod.Ops();mod.rows_at(model,np.array([.5,-1.,2.]),ops);rows=L.report();assert rows['named_numeric_total']==350,rows
    mod,L,_=load();ops=mod.Ops();mod.State(model,np.ones(3),np.eye(3)[0],np.eye(3)[1],np.eye(3)[2],ops);state=L.report();assert state['named_numeric_total']==350+180+45+72,state
    mod,L,_=load();v=mod._Sint(.5,.7,.1,.9);nonzero=L.report();assert nonzero['named_numeric_total']==20,nonzero
    mod,L,_=load();assert mod._Sint(0.,0.,.1,.9)==0.;zero=L.report();assert zero['named_numeric_total']==2,zero
    src='''
def example(a, boom):
    return 2. < a < boom()
def inplace(a):
    a[:,0] += 2.
    return a
def sorting(a):
    a.sort(key=lambda e:e)
    return a
'''
    L=Ledger();scope={'_arithmetic':L};exec(compile(ast.fix_missing_locations(Rewrite().visit(ast.parse(src))),'primitive','exec'),scope)
    assert not scope['example'](1.,lambda:(_ for _ in ()).throw(AssertionError('short circuit lost')))
    assert L.report()['named_numeric_total']==1
    a=np.ones((2,2));assert scope['inplace'](a) is a;assert np.array_equal(a,np.array([[3.,1.],[3.,1.]]))
    assert scope['sorting']([3.,1.,2.])==[1.,2.,3.]
    extra=L.report();assert extra['totals']['array_Add']==2 and extra['totals']['sort_compare']==4,extra
    out={'zero_rows_at':rows,'zero_State':state,'Sint_nonzero':nonzero,'Sint_zero':zero,'shortcircuit_inplace_sort':extra}
    Path(__file__).with_name('arithmetic_primitive_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v['named_numeric_total'] for k,v in out.items()}))
if __name__=='__main__':main()
