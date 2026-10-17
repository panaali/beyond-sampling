"""Frozen analytic primitive and unchanged-estimator parity controls."""
import ast,json,time,traceback
from pathlib import Path
import numpy as np
import trace as T
HERE=Path(__file__).resolve().parent

def expression(source,env,expected):
 ledger=T.Ledger();tree=ast.parse('def fixture():\n '+source.replace('\n','\n '));ns=dict(np=np,_arithmetic=ledger,**env)
 exec(compile(ast.fix_missing_locations(T.Rewrite().visit(tree)),'analytic_fixture','exec'),ns)
 value=ns['fixture']();report=ledger.report();assert report['named_numeric_total']==expected,(source,report['totals'],expected)
 return dict(source=source,value=np.asarray(value).tolist(),expected_numeric=expected,ledger=report)

def main():
 start=time.process_time();ops=[]
 ops.append(expression('return a @ b',dict(a=np.array([1.,0.,0.]),b=np.array([0.,1.,0.])),5))
 ops.append(expression('return np.cross(a,b)',dict(a=np.array([1.,0.,0.]),b=np.array([0.,1.,0.])),9))
 ops.append(expression('return np.linalg.norm(a)',dict(a=np.array([3.,4.,0.])),6))
 ops.append(expression('return a @ b',dict(a=np.arange(6.).reshape(2,3),b=np.arange(6.).reshape(3,2)),20))
 ops.append(expression('return np.abs(a).max(0)',dict(a=np.arange(6.).reshape(2,3)),9))
 ops.append(expression('return np.mean(a,0)',dict(a=np.arange(6.).reshape(2,3)),6))
 ops.append(expression('return a.sum()',dict(a=np.arange(6.)),5))
 ops.append(expression('return (a != 0.)',dict(a=np.array([0.,1.,-1.])),3))
 ops.append(expression('return min(x for x in a)',dict(a=[2.,-1.,3.]),2))
 ops.append(expression('return 0. < a < b',dict(a=-1.,b=3.),1))
 a=np.array([1.,2.]);ops.append(expression('a=x\nb=a\na += c\nreturn a is b',dict(x=a,c=np.array([3.,4.])),2));assert np.array_equal(a,[4.,6.])
 rows=[]
 for name,wi,wh,wo in [('zero',[0.],[[0.]],[[0.],[0.],[0.]]),('zero_readout',[1.],[[0.]],[[0.],[0.],[0.]]),('identical_readout',[1.],[[0.]],[[1.],[1.],[1.]]),('dominant_first',[1.],[[0.]],[[1.],[0.],[-1.]]),('mixed',[1.,-1.],[[0.,1.],[1.,0.]],[[1.,0.],[0.,1.],[-1.,0.]])]:
  model=tuple(np.asarray(x) for x in (wi,wh,wo));baseline=T.evaluate(model,False);traced=T.evaluate(model,True)
  for key in ('hex','legacy','stats','info'):assert baseline[key]==traced[key],(name,key,baseline[key],traced[key])
  rows.append(dict(name=name,baseline=baseline,traced=traced))
 out=dict(operators=ops,models=rows,cpu=time.process_time()-start,scope='Accounting controls preserve existing source errors; no accuracy repair or full native cost.')
 (HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(cpu=out['cpu'],operator_cases=len(ops),models=[dict(name=r['name'],value=r['traced']['value'],legacy=r['traced']['legacy'],numeric=r['traced']['ledger']['named_numeric_total']) for r in rows])))
if __name__=='__main__':
 try:main()
 except Exception as exc:
  (HERE/'controls_failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2)+'\n');raise
