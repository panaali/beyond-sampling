"""Eight frozen transparent-adapter controls, before any _orth5 invocation."""
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from trace import Trace,transform

HERE=Path(__file__).resolve().parent
OUT=HERE/'controls.json'

def bits(x):
    if isinstance(x,np.ndarray):return {'shape':list(x.shape),'dtype':str(x.dtype),'bytes':x.tobytes().hex()}
    if isinstance(x,(bool,int)):return {'type':type(x).__name__,'value':x}
    if isinstance(x,(float,np.floating)):return {'type':type(x).__name__,'hex':float(x).hex()}
    if isinstance(x,(list,tuple)):return [bits(v) for v in x]
    raise TypeError(type(x))

def main():
    assert not OUT.exists(),'Preserve earlier controls'
    start=time.process_time();rows=[]
    fixtures=[
      ('signed_scalar','def f(a,b):\n acc=a\n acc+=b\n return (a+b,a-b,a*b,a/b,-a,+b,a**2,acc)\n',(-0.0,2.0),
       {'numeric_expression:Add':2,'numeric_expression:Sub':1,'numeric_expression:Mult':1,'numeric_expression:Div':1,'numeric_expression:USub':1,'numeric_expression:UAdd':1,'opaque_power:Pow':1}),
      ('integer_sign','def f(a,b):\n return ((-1)**a,a+b,a!=b)\n',(2,3),
       {'integer_expression:USub':1,'integer_expression:Pow':1,'integer_expression:Add':1,'integer_comparison:NotEq':1}),
      ('broadcast','def f(a,b):\n return (a+b,a*b)\n',(np.array([[1.],[2.]]),np.array([[3.,4.,5.]])),
       {'numeric_expression:Add':6,'numeric_expression:Mult':6}),
      ('dot','def f(a,b):\n return a@b\n',(np.array([1.,2.,3.]),np.array([4.,5.,6.])),{'opaque_matmul:MatMult':1}),
      ('matrix','def f(a,b):\n return a@b\n',(np.array([[1.,2.,3.],[4.,5.,6.]]),np.array([[1.,2.],[3.,4.],[5.,6.]])),{'opaque_matmul:MatMult':4}),
      ('sqrt_maximum','def f(a):\n return np.sqrt(np.maximum(a,0.0))\n',(np.array([-1.,4.,9.]),),{'opaque_call:np.maximum':3,'opaque_call:np.sqrt':3}),
      ('short_circuit','def f(a,mark):\n return (a<0 and mark(),a>0 or mark())\n',None,{'numeric_comparison:Lt':2,'numeric_comparison:Gt':2}),
      ('builtin_libm','def f(a):\n return (sum(a),min(a),abs(a[0]),math.atan2(a[0],a[1]))\n',([-1.,2.,3.],),
       {'opaque_call:sum':1,'opaque_call:min':1,'opaque_call:abs':1,'opaque_call:math.atan2':1}),
    ]
    out={'complete':False,'rows':rows,'source_sha256':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['trace.py','controls.py','PROTOCOL.md']},'numpy_version':np.__version__,'python_version':sys.version}
    try:
        for name,source,args,expected in fixtures:
            tree=ast.parse(source);tr=Trace();normal={'np':np,'math':math};wrapped={'np':np,'math':math,'__trace':tr}
            exec(compile(tree,'analytic_control','exec'),normal)
            exec(compile(transform(tree),'analytic_control_adapted','exec'),wrapped)
            row={'name':name,'source':source,'expected_category_output_elements':expected};rows.append(row)
            if name=='short_circuit':
                basecalls=[];tracecalls=[]
                def mark_base():basecalls.append(1);return True
                def mark_trace():tracecalls.append(1);return True
                baseline=[normal['f'](a,mark_base) for a in [-1.0,1.0]]
                value=[wrapped['f'](a,mark_trace) for a in [-1.0,1.0]]
                row['side_effect_counts']=[len(basecalls),len(tracecalls)]
                assert row['side_effect_counts']==[2,2]
            else:baseline=normal['f'](*args);value=wrapped['f'](*args)
            row.update(baseline=bits(baseline),adapted=bits(value),trace=tr.serial())
            assert row['baseline']==row['adapted'],name
            assert dict(tr.counts)==expected,(name,dict(tr.counts),expected)
        out['complete']=True
    except BaseException as e:out['failure']=repr(e);raise
    finally:
        out['cpu_seconds']=time.process_time()-start
        out['orth5_model_MC_calls']=0
        OUT.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'complete':out['complete'],'controls':len(rows),'cpu_seconds':out['cpu_seconds']}))

if __name__=='__main__':main()
